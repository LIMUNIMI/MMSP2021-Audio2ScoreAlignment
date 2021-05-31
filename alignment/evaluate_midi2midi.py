"""
A module to evaluate and compare midi-to-midi alignment methods
"""

import os
import pickle
from dataclasses import dataclass
from time import time
from typing import List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import wilcoxon
from tqdm import tqdm, trange

from . import alignment_eita, alignment_fede, alignment_tafe
from . import settings as s
from .asmd.asmd import asmd, dataset_utils
from .utils import onsets_clusterize, parallel_monitor

mlflow.set_experiment("midi2midi")
mlflow.start_run()


@dataclass
class Result():

    name: str
    time: float
    index: int
    num_notes: int
    missing_precision: float
    missing_recall: float
    missing_fmeasure: float
    extra_precision: float
    extra_recall: float
    extra_fmeasure: float
    match_precision: float
    match_recall: float
    match_fmeasure: float


class MissingExtraManager():
    def __init__(self, dataset, i):
        self.index = i
        self.score, self.extra = dataset_utils.get_score_mat(
            dataset, i, score_type=['misaligned'], return_notes='extra')

        self.perfm, self.missing = dataset_utils.get_score_mat(
            dataset,
            i,
            score_type=['precise_alignment', 'broad_alignment'],
            return_notes='missing')
        self.num_notes = self.perfm.shape[0]
        self.noextra_idx = np.nonzero(~self.extra)[0]
        self.nomissing_idx = np.nonzero(~self.missing)[0]
        self.matched = np.logical_and(~self.extra, ~self.missing)
        self.rng = np.random.default_rng(1917)

    def get_score(self):
        """
        Return a real-world score with missing/extra notes and onsets
        clusterized with minimum th chosen with uniform random distribution in
        (0.02, 0.07)
        """
        # get the score
        score = self.score[self.noextra_idx]
        onsets_clusterize(score, self.rng)
        return score

    def get_perfm(self):
        """
        Return a real-world performance with missing/extra notes
        """
        return self.perfm[self.nomissing_idx]

    def get_gt(self):
        """
        Return ground-truth of onsets and offsets relative to the notes in the
        score returned by `get_score`
        """
        return self.perfm[self.noextra_idx, 1:3]

    def evaluate(self, match: np.ndarray, *args) -> Result:
        """
        Takes a `match` between real world score and performance and returns
        results
        """
        missing = np.zeros_like(self.missing)
        extra = np.zeros_like(self.extra)
        matched = np.zeros_like(self.matched)
        if match.size > 0:
            score_missing = find_lacking_indices(match[:, 0],
                                                 self.noextra_idx.shape[0])
            missing_idx = self.noextra_idx[score_missing]
            missing[missing_idx] = True

            score_extra = find_lacking_indices(match[:, 1],
                                               self.nomissing_idx.shape[0])
            extra_idx = self.nomissing_idx[score_extra]
            extra[extra_idx] = True

            matched_idx = np.nonzero(
                self.nomissing_idx[match[:, 1]] == self.noextra_idx[match[:,
                                                                          0]])[0]
            matched[matched_idx] = True

        return Result(*args, self.index, self.num_notes,
                      *prf(self.missing, missing), *prf(self.extra, extra),
                      *prf(self.matched, matched))  # type: ignore


def prf(target: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    tp = np.logical_and(target, pred).sum()
    fp = np.logical_and(~target, pred).sum()
    fn = np.logical_and(target, ~pred).sum()

    p = tp / (tp + fp + 1e-32)
    r = tp / (tp + fn + 1e-32)
    f = 2 * tp / (2 * tp + fp + fn + 1e-32)
    return p, r, f


def find_lacking_indices(match: np.ndarray, shape: int) -> np.ndarray:
    """
    Returns a boolean array with `True` in indices that are not present in
    `match`
    """
    out = np.ones(shape, dtype=np.bool8)
    out[match] = False
    return out


def eval_fede(me_manager: MissingExtraManager) -> Result:
    ttt = time()
    score = me_manager.get_score()
    perfm = me_manager.get_perfm()
    fede_match, _, _ = alignment_fede.get_matching_notes(
        score.copy(),
        perfm.copy(),

        # thresholds=[1.0,],)
        score_th=0.0)
    fede_time = time() - ttt
    # fede_match contains dummy initial and end point, removing them...
    if fede_match.size > 0:
        fede_match = fede_match[np.logical_and(fede_match[:, 0] > 0,
                                               fede_match[:, 1] > 0)]
        fede_match = fede_match[np.logical_and(
            fede_match[:, 0] < fede_match[:, 0].max(),
            fede_match[:, 1] < fede_match[:, 1].max())]
    fede_res = me_manager.evaluate(fede_match - 1, 'fede', fede_time)
    return fede_res


def test(
    i: int, dataset: asmd.Dataset
) -> Tuple[Optional[Result], Optional[Result], Optional[np.ndarray]]:

    me_manager = MissingExtraManager(dataset, i)

    # skipping long scores
    if me_manager.num_notes > s.MAX_NOTES or me_manager.num_notes < s.MIN_NOTES:
        return None, None, None

    score = me_manager.get_score()
    ground_truth = me_manager.get_gt()
    perfm = me_manager.get_perfm()

    ttt = time()
    eita_match = parallel_monitor(alignment_eita.get_matching_notes,
                                  rss=s.MAX_MEM,
                                  wtime=s.MAX_TIME,
                                  call_args=(score.copy(), perfm.copy()),
                                  call_kwargs=dict(timeout=None))
    if eita_match is None:
        # skip even fede and tafe for fast evaluation
        return None, None, None

    eita_time = time() - ttt
    eita_res = me_manager.evaluate(eita_match, 'eita', eita_time)

    if s.EVAL_FEDE:
        fede_res: Optional[Result] = eval_fede(me_manager)
    else:
        fede_res = None

    if dataset.get_score_duration(i) < s.MAX_DURATION:
        if s.EVAL_FEDE:
            fede_alignment = alignment_fede.fede_align(score.copy(),
                                                       perfm.copy(),
                                                       score_th=0.0)
        eita_alignment = parallel_monitor(alignment_eita.eita_align,
                                          rss=s.MAX_MEM,
                                          wtime=s.MAX_TIME,
                                          call_args=(score.copy(),
                                                     perfm.copy()))
        tafe_alignment = parallel_monitor(alignment_tafe.tafe_align,
                                          rss=s.MAX_MEM,
                                          wtime=s.MAX_TIME,
                                          call_args=(score.copy(),
                                                     perfm.copy()))
        if s.EVAL_FEDE:
            fede_err = np.abs(ground_truth - np.stack(fede_alignment, axis=1))
        eita_err = np.abs(ground_truth - np.stack(eita_alignment, axis=1))
        tafe_err = np.abs(ground_truth - np.stack(tafe_alignment, axis=1))
        if s.EVAL_FEDE:
            err = np.stack([fede_err, eita_err, tafe_err])
        else:
            err = np.stack([eita_err, tafe_err])
    else:
        err = None

    return eita_res, fede_res, err


def view_fig(fig, name=None):
    try:
        mlflow.log_figure(fig, str(time()) + '.html' if not name else name)
    except Exception:
        fig.write_image(fig.layout.title.text.replace(" ", "_") + '.svg')


def evaluate(df: pd.DataFrame, err: pd.DataFrame):
    # For now, just compute the averages
    df_eita = df.loc[df['name'] == 'eita']
    if s.EVAL_FEDE:
        df_fede = df.loc[df['name'] == 'fede']

    def eval_kind(kind: str):
        # precision. recall, f-measure
        if s.EVAL_FEDE:
            fede_prec = df_fede[kind + '_precision']
            fede_rec = df_fede[kind + '_recall']
            fede_f1s = df_fede[kind + '_fmeasure']
        eita_prec = df_eita[kind + '_precision']
        eita_rec = df_eita[kind + '_recall']
        eita_f1s = df_eita[kind + '_fmeasure']

        # p-values
        if s.EVAL_FEDE:
            prec_p = wilcoxon(fede_prec, eita_prec)
            rec_p = wilcoxon(fede_rec, eita_rec)
            f1s_p = wilcoxon(fede_f1s, eita_f1s)

        # printing
        print(f"Eita prec: {eita_prec.mean():.2e}, {eita_prec.std():.2e}")
        if s.EVAL_FEDE:
            print(f"Fede prec: {fede_prec.mean():.2e}, {fede_prec.std():.2e}")
            print(f"Pval prec: {prec_p[1]:.2e}, {prec_p[0]:.2e}")
            print("---")
        print(f"Eita rec: {eita_rec.mean():.2e}, {eita_rec.std():.2e}")
        if s.EVAL_FEDE:
            print(f"Fede rec: {fede_rec.mean():.2e}, {fede_rec.std():.2e}")
            print(f"Pval rec: {rec_p[1]:.2e}, {rec_p[0]:.2e}")
            print("---")
        print(f"Eita f1s: {eita_f1s.mean():.2e}, {eita_f1s.std():.2e}")
        if s.EVAL_FEDE:
            print(f"Fede f1s: {fede_f1s.mean():.2e}, {fede_f1s.std():.2e}")
            print(f"Pval f1s: {f1s_p[1]:.2e}, {f1s_p[0]:.2e}")

        # plotting
        fig = px.violin(
            df,
            y=[kind + '_precision', kind + '_recall', kind + '_fmeasure'],
            box=True,
            color='name',
            violinmode='group',
            range_y=[0, 1],
            title=kind + " notes")
        # layouts
        for data in fig.data:
            data.meanline = dict(visible=True, color='white', width=1)
            data.box.line.color = 'white'
            data.box.line.width = 1
        # show/save
        view_fig(fig, kind + '.html')

    print("\nMissing notes")
    eval_kind('missing')
    print("\nExtra notes")
    eval_kind('extra')
    print("\nMatch")
    eval_kind('match')

    print("\nTime")
    eita_time = df_eita['time']
    if s.EVAL_FEDE:
        fede_time = df_fede['time']
        time_p = wilcoxon(fede_time, eita_time)

    print(f"Eita time: {eita_time.mean():.2e}, {eita_time.std():.2e}")
    if s.EVAL_FEDE:
        print(f"Fede time: {fede_time.mean():.2e}, {fede_time.std():.2e}")
        print(f"Pval time: {time_p[1]:.2e}, {time_p[0]:.2e}")
    # time scatter-plot
    view_fig(
        px.scatter(df,
                   x='num_notes',
                   y='time',
                   color='name',
                   log_y=True,
                   trendline='lowess',
                   trendline_color_override='black'), 'time.html')

    # plotting thresholds
    for col in err['col'].unique():
        fig = px.line(err[err['col'] == col],
                      x='th',
                      y='val',
                      color='row',
                      line_group='row',
                      title=col + ' thresholds')
        view_fig(fig, f'thresholds_{col}.html')


def make_dataframe(
    results: List[Tuple[Optional[Result], Optional[Result],
                        Optional[np.ndarray]]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _errs = []
    data = []
    eita_counter = 0
    fede_counter = 0
    tafe_counter = 0
    for res_eita, res_fede, err in results:
        flag = True
        if res_eita is None:
            eita_counter += 1
            flag = False

        if s.EVAL_FEDE:
            if res_fede is None:
                fede_counter += 1
                flag = False

        if flag:
            data.append(vars(res_eita))
            if s.EVAL_FEDE:
                data.append(vars(res_fede))

        if err is not None:
            _errs.append(err)
        else:
            tafe_counter += 1

    print(f"Eita wasn't run on {eita_counter} songs")
    if s.EVAL_FEDE:
        print(f"Fede wasn't run on {fede_counter} songs")
    print(f"Tafe wasn't run on {tafe_counter} songs")

    errs = pd.DataFrame()
    if len(_errs) > 0:
        _errs = np.concatenate(_errs, axis=1)
        if s.EVAL_FEDE:
            _methods = ['fede', 'eita', 'tafe']
        else:
            _methods = ['eita', 'tafe']
        print("Filling thresholds")
        for th in tqdm(np.arange(0, 1, 0.01)):
            for col in [0, 1, slice(0, 2)]:
                for method in range(len(_errs)):  # type: ignore
                    analyzed = _errs[method, :, col]  # type: ignore
                    errs = errs.append(
                        dict(th=th,
                             col=str(col),
                             row=_methods[method],
                             val=float(np.count_nonzero(analyzed < th)) /
                             analyzed.size),
                        ignore_index=True)
    return pd.DataFrame(data), errs


def main():
    print("Evaluating midi-to-midi methods")
    alignment_fede.include_julia("alignment/alignment_fede.jl")
    dataset = asmd.Dataset()
    fname = "midi2midi_result.pkl"
    ###
    # res = pickle.load(open("res_" + fname, "rb"))
    # df, err = make_dataframe(res)
    # pickle.dump((df, err), open(fname, "wb"))
    ####
    if os.path.exists(fname):
        df, err = pickle.load(open(fname, "rb"))
    else:
        res = []
        for i in trange(len(dataset)):
            res.append(test(i, dataset))
        pickle.dump(res, open("res_" + fname, "wb"))
        df, err = make_dataframe(res)
        pickle.dump((df, err), open(fname, "wb"))
    evaluate(df, err)


if __name__ == "__main__":
    main()
