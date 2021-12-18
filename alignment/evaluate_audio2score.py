"""
A module to evaluate and compare midi-to-midi alignment methods
"""

import argparse
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Tuple

import mir_eval
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from . import settings as s
from . import utils
from .align import align_to_midi
from .alignment_eita import eita_align
from .alignment_fede import fede_align
from .alignment_tafe import tafe_align
from .asmd.asmd import asmd, dataset_utils
from .asmd.asmd.conversion_tool import generate_missing_extra
from .seba.align import audio_to_score_alignment
from .transcription import transcribe

rng = np.random.default_rng(1992)

# tensorflow *must* be imported after than pytorch, because tensorflow doesn't
# leave space to other frameworks. So bytedance must be first because it uses
# pytorch. Also, cannot import them in global scope because tensorflow breaks
# for old CPU (don't know why in omnizart/magenta doesn't break)
# N.B. since the monitoring happens in separate process and cuda is unloaded
# when the process exit, this is not a problem anymore
# TRANSCR_METHODS = ['magenta', 'omnizart', 'bytedance']
TRANSCR_METHODS = ['omnizart', 'bytedance']

# seba MUST be the last one...
if s.EVAL_FEDE:
    ALIGN_METHODS = ['fede', 'eife', 'tafe', 'seba']
else:
    ALIGN_METHODS = ['eife', 'tafe', 'seba']

MAX_TH = 1.0


def transcr_eval(targ, pred):
    """
    Use package `mir_eval` to evaluate a transcription
    """
    targ = utils.mat2mir_eval(targ)
    pred = utils.mat2mir_eval(pred)
    t1, p1, v1 = targ
    t2, p2, v2 = pred
    # remove initial silence
    t1 -= np.min(t1)
    t2 -= np.min(t2)

    p, r, f, avg = mir_eval.transcription.precision_recall_f1_overlap(
        t1, p1, t2, p2, offset_ratio=None)
    return p, r, f, avg


def collect_statistics(mat, audio, sr):
    """
    Collect various statistics about a song, namely returns:

    * duration of audio
    * number of notes in mat
    * average polyphony in mat
    """
    # import visdom
    # vis = visdom.Visdom()
    # vis.audio(audio[:sr * 20], opts=dict(sample_frequency=sr))

    duration = audio.shape[-1] / sr
    _pr = utils.make_pianoroll(mat, res=0.0625, velocities=False)

    # vis.heatmap(_pr[:, :int(20 / 0.0625)])
    pr = _pr.sum(axis=0)

    # print(pr.shape)
    polyphony = np.mean(pr[pr > 0])

    # __import__('ipdb').set_trace()
    return dict(dur=duration, notes=mat.shape[0], poly=polyphony)


@dataclass
class Evaluator:
    i: int
    dataset: asmd.Dataset
    missing: bool
    counters: Dict[str, int]
    evaluable: bool = True
    results: pd.DataFrame = pd.DataFrame()

    def load_data(self):
        self.score, self.extra = dataset_utils.get_score_mat(
            self.dataset,
            self.i,
            score_type=['misaligned'],
            return_notes='extra')
        # skipping long scores
        if self.score.shape[0] > s.MAX_NOTES or dataset.get_score_duration(
                self.i) > s.MAX_DURATION:
            self.evaluable = False
            self.counters['evaluable'][self.i] = False
            return
        else:
            self.counters['evaluable'][self.i] = True

        self.score[:, 1:3] -= np.min(self.score[:, 1:3])

        self._ground_truth = dataset_utils.get_score_mat(
            self.dataset,
            self.i,
            score_type=['precise_alignment', 'broad_alignment'])
        self._ground_truth[:, 1:3] -= np.min(self._ground_truth[:, 1:3])
        self.ground_truth = self._ground_truth[:, 1:3]

        self.score = utils.onsets_clusterize(self.score, rng)

        self.audio, self.sr = self.dataset.get_audio(self.i)
        if self.missing:
            self.score = self.score[~self.extra]
            self.ground_truth = self.ground_truth[~self.extra]
        self.statistics = collect_statistics(self._ground_truth, self.audio,
                                             self.sr)
        self.transcr_statistics = {}

    def run_seba(self):
        # N.B. seba method is tested without missing notes (only extra notes)!
        seba_alignment = utils.parallel_monitor(
            audio_to_score_alignment,
            rss=s.MAX_MEM,
            wtime=s.MAX_TIME,
            call_args=(np.copy(self.score), (np.copy(self.audio), self.sr)))
        if seba_alignment is None:
            print("Wrong seba!!")
            self.counters['seba'][self.i] = False
            return
        else:
            self.counters['seba'][self.i] = True
            seba_err = np.abs(self.ground_truth -
                              np.stack(seba_alignment, axis=1))

        # recording results
        L = seba_err.shape[0]
        self.results = self.results.append(
            pd.DataFrame(data=dict(song=[self.i] * L,
                                   ons=seba_err[:, 0],
                                   offs=seba_err[:, 1],
                                   method=['seba'] * L)),
            ignore_index=True)

    def transcribe(self, transcr_method):
        try:
            ttt = time()
            self.perfm = utils.parallel_monitor(transcribe,
                                                rss=s.MAX_MEM,
                                                wtime=s.MAX_TIME,
                                                call_args=(np.copy(self.audio),
                                                           self.sr,
                                                           transcr_method))
            self.transcr_time = time() - ttt
        except RuntimeError:
            self.perfm = None
            return

        if self.perfm is None:
            self.transcr_err = True
            self.transcr_method = "error transcription"
            self.counters[transcr_method][self.i] = False
            return

        self.transcr_err = False
        self.counters[transcr_method][self.i] = True
        self.transcr_method = transcr_method
        self.perfm[:, 1:3] -= np.min(self.perfm[:, 1:3])

        # evaluate and record the onset f-measure
        self.transcr_statistics[transcr_method] = transcr_eval(
            self._ground_truth, self.perfm)[2]
        if self.missing:
            # generate random missing notes
            missing, _ = generate_missing_extra(self.perfm.shape[0])
            self.nomissing = ~missing
            # removing missing notes
            self.perfm = self.perfm[self.nomissing]

    def run_eife(self):
        method = self.transcr_method + ' eife'
        try:
            eife_alignment = utils.parallel_monitor(
                align_to_midi,
                rss=s.MAX_MEM,
                wtime=s.MAX_TIME - self.transcr_time,
                call_args=(np.copy(self.score), np.copy(self.perfm),
                           eita_align, (np.copy(self.audio), self.sr)))
            if eife_alignment is None:
                print("Wrong eife!")
                self.counters[method][self.i] = False
                return
            else:
                self.counters[method][self.i] = True
                eife_err = np.abs(self.ground_truth - eife_alignment[:, 1:3])
        except RuntimeError:
            self.counters[method][self.i] = False
            return

        # recording results
        L = eife_err.shape[0]
        self.results = self.results.append(
            pd.DataFrame(data=dict(song=[self.i] * L,
                                   ons=eife_err[:, 0],
                                   offs=eife_err[:, 1],
                                   method=[method] * L)),
            ignore_index=True)

    def run_tafe(self):
        method = self.transcr_method + ' tafe'
        tafe_alignment = utils.parallel_monitor(
            align_to_midi,
            rss=s.MAX_MEM,
            wtime=s.MAX_TIME - self.transcr_time,
            call_args=(np.copy(self.score), np.copy(self.perfm), tafe_align))
        if tafe_alignment is None:
            print("Wrong Tafe!")
            self.counters[method][self.i] = False
            return
        else:
            self.counters[method][self.i] = True
            tafe_err = np.abs(self.ground_truth - tafe_alignment[:, 1:3])

        # recording results
        L = tafe_err.shape[0]
        self.results = self.results.append(
            pd.DataFrame(data=dict(song=[self.i] * L,
                                   ons=tafe_err[:, 0],
                                   offs=tafe_err[:, 1],
                                   method=[method] * L)),
            ignore_index=True)

    def run_fede(self):
        method = self.transcr_method + ' fede'
        fede_alignment = align_to_midi(np.copy(self.score),
                                       np.copy(self.perfm),
                                       fede_align,
                                       score_th=None,
                                       audio_data=(np.copy(self.audio),
                                                   self.sr))
        if fede_alignment is None:
            print("Wrong Fede!!")
            self.counters[method][self.i] = False
            return
        else:
            self.counters[method][self.i] = True
            fede_err = np.abs(self.ground_truth - fede_alignment[:, 1:3])

        # recording results
        L = fede_err.shape[0]
        self.results = self.results.append(
            pd.DataFrame(data=dict(song=[self.i] * L,
                                   ons=fede_err[:, 0],
                                   offs=fede_err[:, 1],
                                   method=[method] * L)),
            ignore_index=True)


def test(
        i: int, dataset: asmd.Dataset, missing: bool,
        counters: Dict[str, int]) -> Optional[Tuple[pd.DataFrame, dict, dict]]:

    evaluator = Evaluator(i, dataset, missing, counters)
    evaluator.load_data()
    if not evaluator.evaluable:
        return None
    evaluator.run_seba()

    # for transcr_method in ['bytedance']
    for transcr_method in TRANSCR_METHODS:
        print(">>> Working with " + transcr_method)
        evaluator.transcribe(transcr_method)
        if not evaluator.transcr_err:
            if s.EVAL_FEDE:
                evaluator.run_fede()
            evaluator.run_eife()
            evaluator.run_tafe()

    return evaluator.results, evaluator.statistics, evaluator.transcr_statistics


__MAKER_DATASET_LEN = 0


def __maker():
    return [None] * __MAKER_DATASET_LEN


def dataset_test(dataset):
    res = pd.DataFrame()
    stats = pd.DataFrame()
    transcr_stats = pd.DataFrame()

    # must be global, otherwise we cannot pickle the result...
    global __MAKER_DATASET_LEN
    __MAKER_DATASET_LEN = len(dataset)

    counters: Dict[str, List[Optional[bool]]] = defaultdict(__maker)
    for i in trange(len(dataset)):
        out = test(i, dataset, args.missing, counters)
        if out is not None:
            res = res.append(out[0], ignore_index=True)
            stats = stats.append(out[1], ignore_index=True)
            transcr_stats = transcr_stats.append(out[2], ignore_index=True)
    return res, stats, transcr_stats, counters


def view_fig(fig, name=None):
    try:
        # don't know why but whithout the following line mlflow breaks
        import matplotlib.figure  # noqa: autoimport
        mlflow.log_figure(fig, str(time()) + '.html' if not name else name)
    except Exception:
        fig.write_image(fig.layout.title.text.replace(" ", "_") + '.svg')


def check_counters(counters, idx):
    for key, val in counters.items():
        if key != 'evaluable':
            if not val[idx]:
                return False
    return True


def purge_df(err, counters):

    out = pd.DataFrame()
    for song in err['song'].unique():
        if check_counters(counters, song):
            out = out.append(err[err['song'] == song], ignore_index=True)
    return out


def exclude_methods(res, exclude=['magenta']):
    for method in res['method'].unique():
        if method in exclude:
            res = res[res['method'] != method]
    return res


def micro_benchmark(err):
    out = pd.DataFrame()
    for method in err['method'].unique():
        _err = err[err['method'] == method]
        L = _err.shape[0]
        for th in np.arange(0, MAX_TH, 0.01):
            ons = np.count_nonzero(_err['ons'] < th) / L
            offs = np.count_nonzero(_err['offs'] < th) / L
            both = np.count_nonzero(
                np.logical_and(_err['ons'] < th, _err['offs'] < th)) / L
            out = out.append(dict(method=method,
                                  th=th,
                                  ons=ons,
                                  offs=offs,
                                  both=both),
                             ignore_index=True)
    return out


def _process(_macro, method, th):
    return _macro[(_macro['method'] == method) & (_macro['th'] == th)].mean()


def macro_benchmark(err):
    print("Computing per-song curves...")
    par_res = Parallel(n_jobs=-1)(
        delayed(micro_benchmark)(err[err['song'] == song])
        for song in tqdm(err['song'].unique()))
    _macro = pd.concat(par_res, ignore_index=True)

    # averaging
    print("Averaging curves per-method...")
    macro = pd.DataFrame()
    for method in _macro['method'].unique():
        par_res = Parallel(n_jobs=-1)(delayed(_process)(_macro, method, th)
                                      for th in tqdm(_macro['th'].unique()))
        macro_method = pd.concat(par_res, ignore_index=True, axis=1).T
        macro_method['method'] = method
        macro = macro.append(macro_method, ignore_index=True)
    return macro


def plot_curves(err: pd.DataFrame):

    # plotting thresholds micro-benchmark
    print("Computing micro benchmark")
    micro = micro_benchmark(err)
    for col in ['ons', 'offs', 'both']:
        fig = px.line(micro,
                      x='th',
                      y=col,
                      color='method',
                      line_group='method',
                      title='micro ' + col + ' thresholds')
        view_fig(fig, f'micro_thresholds_{col}.html')

    # plotting thresholds macro-benchmark
    print("computing macro benchmarks")
    macro = macro_benchmark(err)
    for col in ['ons', 'offs', 'both']:
        fig = px.line(
            macro,
            x='th',
            y=col,
            color='method',
            # error_y='std_u',
            # error_y_minus='std_l',
            line_group='method',
            title='macro ' + col + ' thresholds')
        # fig = plotly_error_bands(fig)
        view_fig(fig, f'macro_thresholds_{col}.html')


def count(counters: Dict[str, List[Optional[bool]]], fname=None):
    if fname:
        f = open(fname, "wt")
    for key, items in counters.items():
        line = f"{key}: {sum([1 for i in items if i is True])}"
        print(line)
        if fname:
            f.write(line + '\n')
        line = f"not {key}: {sum([1 for i in items if i is False])}"
        print(line)
        if fname:
            f.write(line + '\n')
    if fname:
        f.close()


def plot_stats(stats, transcr_stats):
    print("Plotting stats...")

    def _plot(*args, **kwargs):
        fig = px.violin(*args, **kwargs)

        # layouts
        for data in fig.data:
            data.meanline = dict(visible=True, color='white', width=1)
            data.box.line.color = 'white'
            data.box.line.width = 1
        title = kwargs.get('title') or str(time())
        view_fig(fig, "_".join(title.split(" ")) + ".html")

    _plot(stats, x=None, y='dur', title='Duration')
    _plot(stats, x=None, y='notes', title='Num notes')
    _plot(stats, x=None, y='poly', title='Polyphony')
    _plot(transcr_stats, title='f1measure', range_y=[0, 1])


if __name__ == "__main__":
    # CLI arguments
    argparser = argparse.ArgumentParser("Evaluate audio2midi")
    argparser.add_argument(
        "-m",
        "--missing",
        action='store_true',
        help=
        'If used, missing/extra notes will be simulated; In such case, the evaluation will try to align the notes not present in the performance but not the notes removed from score. We still cannot count missing/extra notes identification ability because the transcription also has missing/extra notes. Note that seba method is tested with extra notes only (cannot remove notes from audio)'
    )
    argparser.add_argument(
        "-p",
        "--piano",
        action='store_true',
        help=
        'If used, only music with solo piano is used for evaluation, otherwise only music that is not solo piano is used'
    )
    argparser.add_argument(
        "-o",
        "--overfit",
        action='store_true',
        help=
        'If used, Maestro and MuscNet are removed from the datasets to avoid overfit'
    )
    argparser.add_argument("-d",
                           "--dataset",
                           action='store',
                           type=str,
                           nargs=1,
                           default=None,
                           help='If used, only the specified dataset is used.')
    args = argparser.parse_args()

    # set-up mlflow
    mlflow.set_experiment("audio2midi_mmsp")
    run_name = 'piano' if args.piano else 'multi'
    run_name += '-missing' if args.missing else '-nomissing'
    run_name += '-' + args.dataset[0].lower() if args.dataset else '-all'
    mlflow.start_run(run_name=run_name)

    # preparing dataset
    print("Evaluating audio-to-midi methods")
    dataset = asmd.Dataset()
    if args.overfit:
        dataset = dataset_utils.filter(dataset,
                                       datasets=['MuscNet', 'Maestro'])
        dataset = dataset_utils.complement(dataset)

    dataset = dataset_utils.filter(dataset, instruments=['piano'])

    if not args.piano:
        dataset = dataset_utils.complement(dataset)

    if args.dataset:
        dataset = dataset_utils.filter(dataset, datasets=args.dataset)

    # Loading result if exist
    fname = run_name + 'a2m.pkl'
    if os.path.exists(fname):
        out = pickle.load(open(fname, "rb"))
    else:
        out = dataset_test(dataset)
        pickle.dump(out, open(fname, "wb"))

    res, stats, transcr_stats, counters = out
    print("Errors:")
    count(counters, fname=run_name + '.notes')

    # plotting
    # exclude_list = ['magenta'] if not args.piano else []
    res = exclude_methods(res, exclude=[])
    res = purge_df(res, counters)
    tot_songs = res['song'].unique().shape[0]
    print(f"After purging, evaluation is run on {tot_songs} songs")
    plot_curves(res)
    plot_stats(stats, transcr_stats)
