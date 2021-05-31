import os
import os.path
import pickle
import random
from copy import deepcopy
from random import choices, uniform
from typing import List, Tuple

import numpy as np
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler, minmax_scale

from .asmd import Dataset
from .conversion_tool import fix_offsets
from .dataset_utils import choice, filter, get_score_mat, union
from .eita.alignment_eita import get_matching_notes
from .idiot import THISDIR
from .utils import mat_stretch

NJOBS = -1
FILE_STATS = os.path.join(THISDIR, "_alignment_stats.pkl")


# TODO: refactoring: most of the stuffs are repeated twice for onsets and durations
class Stats(object):
    def __init__(self,
                 ons_dev_max=0.2,
                 dur_dev_max=0.2,
                 mean_max_ons=None,
                 mean_max_dur=None):
        self.dur_ratios = []
        self.ons_diffs = []
        self.ons_lengths = []
        self.dur_lengths = []
        self.means_ons = []
        self.means_dur = []
        self.ons_dev = []
        self.dur_dev = []
        self.ons_dev_max = ons_dev_max
        self.dur_dev_max = dur_dev_max
        self.mean_max_ons = mean_max_ons
        self.mean_max_dur = mean_max_dur
        self._song_duration_dev = 1
        self._song_onset_dev = 1
        self._song_mean_ons = 0
        self._song_mean_dur = 0
        self._seed = 1992

    def seed(self):
        """
        Calls `seed` on python `random` and then increments its own seed of one
        """
        random.seed(self._seed)
        self._seed += 1
        return self._seed

    def add_data_to_histograms(self, ons_diffs, dur_ratios):
        """
        Method to add data, then you should still compute histograms
        """
        self.ons_dev.append(np.std(ons_diffs))
        self.dur_dev.append(np.std(dur_ratios))
        self.means_ons.append(np.mean(ons_diffs))
        self.means_dur.append(np.mean(dur_ratios))

        self.ons_diffs += StandardScaler().fit_transform(
            ons_diffs.reshape(-1, 1)).tolist()
        self.dur_ratios += StandardScaler().fit_transform(
            dur_ratios.reshape(-1, 1)).tolist()

        self.ons_lengths.append(len(ons_diffs))
        self.dur_lengths.append(len(dur_ratios))

    def get_random_onset_dev(self, k=1):
        self.seed()
        return _get_random_value_from_hist(self.ons_dev_hist,
                                           k,
                                           max_value=self.ons_dev_max)

    def get_random_duration_dev(self, k=1):
        self.seed()
        return _get_random_value_from_hist(self.dur_dev_hist,
                                           k,
                                           max_value=self.dur_dev_max)

    def get_random_mean_ons(self, k=1):
        self.seed()
        return _get_random_value_from_hist(self.means_hist_ons,
                                           k,
                                           max_value=self.mean_max_ons)

    def get_random_mean_dur(self, k=1):
        self.seed()
        return _get_random_value_from_hist(self.means_hist_dur,
                                           k,
                                           max_value=self.mean_max_dur)

    def new_song(self):
        """
        Prepare this object for a new song
        """
        self.seed()
        self._song_duration_dev = self.get_random_duration_dev()
        self.seed()
        self._song_onset_dev = self.get_random_onset_dev()
        self.seed()
        self._song_mean_ons = self.get_random_mean_ons()
        self._song_mean_dur = self.get_random_mean_dur()

    def fill_stats(self, dataset: Dataset):
        """
        Fills this object with data from `datasets`
        """

        global process_

        def process_(i, dataset):
            try:
                score, aligned = get_matching_scores(dataset, i)
            except RuntimeError:
                # skipping if we cannot match the notes for this score
                return None

            # computing diffs
            ons_diffs = score[:, 1] - aligned[:, 1]
            dur_ratios = (aligned[:, 2] - aligned[:, 1]) / (score[:, 2] -
                                                            score[:, 1])
            return ons_diffs, dur_ratios

        # puts in `self._data` onset and duration diffs
        self._data = dataset.parallel(
            process_,  # type: ignore
            n_jobs=NJOBS,
            backend="multiprocessing")

        count = 0
        for res in self._data:
            if res is not None:
                count += 1
                ons_diffs, dur_ratios = res
                self.add_data_to_histograms(ons_diffs, dur_ratios)

        print(
            f"Using {count / len(self._data):.2f} songs ({count} / {len(self._data)})"
        )

    def get_random_durations(self, aligned_dur):
        aligned_dur = np.asarray(aligned_dur)
        self.seed()
        new_dur_ratio = self.get_random_duration_ratio(
            k=len(aligned_dur)) * self._song_duration_dev + self._song_mean_dur
        return aligned_dur / np.abs(new_dur_ratio)

    def get_random_onsets(self, aligned):
        aligned = np.asarray(aligned)
        self.seed()
        new_ons_diff = self.get_random_onset_diff(
            k=len(aligned)) * self._song_onset_dev + self._song_mean_ons

        new_ons = np.sort(aligned + new_ons_diff)
        new_ons -= new_ons.min()
        return new_ons

    def get_random_offsets(self, aligned_ons, aligned_offs, new_ons):
        aligned_ons = np.asarray(aligned_ons)
        aligned_offs = np.asarray(aligned_offs)
        new_ons = np.asarray(new_ons)
        new_dur = self.get_random_durations(aligned_offs - aligned_ons)
        return new_ons + new_dur

    def get_random_onset_diff(self, k=1):
        pass

    def get_random_duration_ratio(self, k=1):
        pass

    def train_on_filled_stats(self):
        """
        Compute all the histograms in tuples (histogram, bin_edges):
        self.means_hist
        self.ons_dev_hist
        self.dur_dev_hist
        """
        self.means_hist_ons = np.histogram(self.means_ons,
                                           bins='auto',
                                           density=True)
        self.means_hist_dur = np.histogram(self.means_dur,
                                           bins='auto',
                                           density=True)
        self.ons_dev_hist = np.histogram(self.ons_dev,
                                         bins='auto',
                                         density=True)
        self.dur_dev_hist = np.histogram(self.dur_dev,
                                         bins='auto',
                                         density=True)


class HistStats(Stats):
    def __init__(self, ons_max=None, dur_max=None, stats: Stats = None):
        super().__init__()
        if stats:
            self.__dict__.update(deepcopy(stats.__dict__))
        self.ons_max = ons_max
        self.dur_max = dur_max

    def train_on_filled_stats(self):
        super().train_on_filled_stats()
        # computing onset and duration histograms
        self.ons_hist = np.histogram(self.ons_diffs, bins='auto', density=True)
        self.dur_hist = np.histogram(self.dur_ratios,
                                     bins='auto',
                                     density=True)

    def get_random_onset_diff(self, k=1):
        self.seed()
        return _get_random_value_from_hist(self.ons_hist,
                                           k,
                                           max_value=self.ons_max)

    def get_random_duration_ratio(self, k=1):
        self.seed()
        return _get_random_value_from_hist(self.dur_hist,
                                           k,
                                           max_value=self.dur_max)

    def __repr__(self):
        return str(type(self))


class HMMStats(Stats):
    def __init__(self, stats: Stats = None):
        super().__init__()

        if stats:
            self.__dict__.update(deepcopy(stats.__dict__))

        n_iter = 100  # maximum number of iterations
        tol = 0.1  # minimum value of log-likelyhood
        covariance_type = 'diag'
        self.onshmm = GMMHMM(
            n_components=20,  # the number of gaussian mixtures
            n_mix=30,  # the number of hidden states
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            verbose=True,
            random_state=self.seed())
        self.durhmm = GMMHMM(
            n_components=2,
            n_mix=3,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            verbose=True,
            random_state=self.seed())

    def get_random_onset_diff(self, k=1):
        x, _state_seq = self.onshmm.sample(k, random_state=self.seed())
        return x[:, 0]

    def get_random_duration_ratio(self, k=1):
        x, _state_seq = self.durhmm.sample(k, random_state=self.seed())
        return x[:, 0]

    def train_on_filled_stats(self):
        super().train_on_filled_stats()

        # train the hmms
        def train(hmm, data, lengths):
            hmm.fit(data, lengths)
            if (hmm.monitor_.converged):
                print("hmm converged!")
            else:
                print("hmm did not converge!")

        print("Training duration hmm...")
        train(self.durhmm, self.dur_ratios, self.dur_lengths)
        print("Training onset hmm...")
        train(self.onshmm, self.ons_diffs, self.ons_lengths)

    def __repr__(self):
        return str(type(self))


def get_matching_scores(dataset: Dataset,
                        i: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a sub-scores of matching notes between `score` and the mos precisely
    aligned data available for song at index `i`

    Returns aligned, score
    """
    mat_aligned = get_score_mat(
        dataset, i, score_type=['precise_alignment', 'broad_alignment'])
    mat_score = get_score_mat(dataset, i, score_type=['score'])

    # stretch to the same average BPM
    mat_stretch(mat_score, mat_aligned)

    # changing float pitches to nearest pitch
    mat_aligned[:, 0] = np.round(mat_aligned[:, 0])
    mat_score[:, 0] = np.round(mat_score[:, 0])

    # apply Eita method
    matching_notes = get_matching_notes(mat_score, mat_aligned, timeout=20)
    if matching_notes is None:
        raise RuntimeError("Cannot match notes for this score!")
    return mat_score[matching_notes[:, 0]], mat_aligned[matching_notes[:, 1]]


def _get_random_value_from_hist(hist, k=1, max_value=None, hmm=False):
    """
    Given a histogram (tuple returned by np.histogram), returns a random value
    picked with uniform distribution from a bin of the histogram. The bin is
    picked following the histogram distribution. If `max` is specified, the
    histogram is first normalized so that the maximum absolute value is the one
    specified.
    """
    if max_value:
        values = minmax_scale(hist[1], (-abs(max_value), abs(max_value)))
    else:
        values = hist[1]
    start = choices(values[:-1], weights=hist[0], k=k)
    bin_w = abs(values[1] - values[0])
    end = np.array(start) + bin_w
    return np.asarray([uniform(start[i], end[i]) for i in range(len(start))])


def evaluate(dataset: Dataset, stats: List[Stats]):
    """
    Computes classical DTW over all datasets and returns avarage and standard
    deviation of all the DTW distances for each `Stats` object in stats

    This function will also need to install the dtw-python module separately
    """
    global process_

    def process_(i: int, dataset: Dataset, stat: Stats):

        # reset the stats for a new song
        stat.new_song()

        try:
            # take the matching notes in the score
            score, aligned = get_matching_scores(dataset, i)
        except RuntimeError:
            # skipping if cannot match notes
            return -1, -1

        # take random standardized differences
        aligned_diff = stat.get_random_onset_diff(k=score.shape[0])
        song_ons_diff = score[:, 1] - aligned[:, 1]
        # computing meang and dev from the matching notes
        mean = np.mean(song_ons_diff)
        std = np.std(song_ons_diff)

        # computing the estimated ons
        ons = np.sort(aligned[:, 1] + aligned_diff * std + mean)

        # computing estmated offs
        dur_ratios = stat.get_random_duration_ratio(k=score.shape[0])
        song_dur = (aligned[:, 2] - aligned[:, 1])
        song_dur_ratio = song_dur / (score[:, 2] - score[:, 1])
        # computing meang and dev from the matching notes
        mean = np.mean(song_dur_ratio)
        std = np.std(song_dur_ratio)

        # computing the estimated offs
        est_ratios = dur_ratios * std + mean
        new_dur = song_dur / est_ratios
        offs = ons + new_dur

        fix_offsets(ons, offs, score[:, 0])

        # DTW between score and affinely transformed new times
        offs_dist = np.abs(offs - score[:, 2]).mean()
        ons_dist = np.abs(ons - score[:, 1]).mean()
        return ons_dist, offs_dist

    for stat in stats:
        print(f"Evaluating {stat}")
        distances = dataset.parallel(
            process_,  # type: ignore
            stat,
            n_jobs=NJOBS,
            max_nbytes=None,
            backend="multiprocessing")
        # removing scores where we couldn't match notes
        distances = np.asarray(distances)
        valid_scores = np.count_nonzero(distances[:, 0] > 0)
        print(
            f"Used {valid_scores / len(dataset)} scores ({valid_scores} / {len(dataset)})"
        )
        distances = distances[distances[:, 0] >= 0]
        print(f"Statics for {stat} and Onsets")
        print(f"Avg: {np.mean(distances[:, 0]):.2e}")
        print(f"Std {np.std(distances[:, 0]):.2e}")

        print(f"Statics for {stat} and Offsets")
        print(f"Avg: {np.mean(distances[:, 1]):.2e}")
        print(f"Std {np.std(distances[:, 1]):.2e}")


def get_stats(method='histogram', save=True, train=True):
    """
    Computes statistics, histogram, dumps the object to file and returns it
    """
    if os.path.exists(FILE_STATS):
        return pickle.load(open(os.path.join(FILE_STATS), "rb"))

    elif train:
        dataset = _get_dataset()
        print("Computing statistics")
        stats = Stats()
        stats.fill_stats(dataset)
        return _train_model(stats, method, save)
    else:
        return None


def _get_dataset():
    dataset = Dataset()
    # dataset = filter(dataset,
    #                  datasets=['Bach10', 'traditional_flute', 'MusicNet'],
    #                  copy=True)

    dataset = union(
        filter(dataset,
               datasets=[
                   'vienna_corpus', 'Bach10', 'traditional_flute', 'MusicNet'
               ],
               copy=True),
        filter(dataset, datasets=['Maestro'], groups=['asap'], copy=True))
    return dataset


def _train_model(stats: Stats, method: str, save: bool):

    if method == 'histogram':
        stats = HistStats(stats=stats)
    elif method == 'hmm':
        stats = HMMStats(stats=stats)

    stats.train_on_filled_stats()

    if save:
        print("Saving statistical model")
        if os.path.exists(FILE_STATS):
            os.remove(FILE_STATS)
        pickle.dump(stats, open(FILE_STATS, 'wb'))
    return stats


if __name__ == '__main__':
    dataset = _get_dataset()
    print("Computing statistics")
    stats = Stats()
    trainset, testset = choice(dataset,
                               p=[0.7, 0.3],
                               random_state=stats.seed())
    stats.fill_stats(trainset)

    for method in ['hmm', 'histogram']:
        model = _train_model(stats, method, False)
        # stat = pickle.load(
        #     open(os.path.join(THISDIR, "_alignment_stats.pkl"), "rb"))
        evaluate(testset, [
            model,
        ])
