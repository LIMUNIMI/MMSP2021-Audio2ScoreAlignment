import gzip
import inspect
import json
import os
from os.path import join as joinpath
from typing import List, Optional

import numpy as np
from essentia.standard import MetadataReader, Resample
import essentia as es
from joblib import Parallel, delayed
from tqdm import tqdm

from . import utils
from .dataset_utils import chose_score_type, filter
from .idiot import THISDIR

# this only for detecting package directory but breaks readthedocs

# THISDIR = './datasets/'


class Dataset(object):
    def __init__(self,
                 definitions=[joinpath(THISDIR, 'definitions/')],
                 metadataset_path=joinpath(THISDIR, 'datasets.json'),
                 empty=False):
        """
        Load the dataset description and populate the paths

        This object has a fundamental field named `paths` which is a list; each
        entry contain another list of 3 values representing thepath to,
        respectively: mixed recording, signle-sources audio, ground-truth file
        per each source

        Parameters
        ----------

        * definitions : list of str
            paths where `json` dataset definitions are stored; if empty, the
            default definitions are used

        * metadataset_path : str
            the path were the generic information about where this datetimeis
            installed are stored

        * empty : bool
            if True, no definition is loaded

        Returns
        -------
        * AudioScoreDataset :
            instance of the class
        """

        self.datasets = []
        if not empty:
            if len(definitions) == 0:
                definitions = [joinpath(THISDIR, 'definitions/')]
            for path in definitions:
                self.datasets += load_definitions(path)

        # opening medataset json file
        self.metadataset = json.load(open(metadataset_path, 'rt'))
        self.install_dir = self.metadataset['install_dir']
        if self.install_dir.endswith('/'):
            # this shouldn't happen actually...
            self.install_dir = self.install_dir[:-1]

        # self.decompress_path = self.metadataset['decompress_path']
        self.paths = []
        self._chunks = {}

        # let's include all the songs and datasets
        for d in self.datasets:
            d['included'] = True
            for s in d['songs']:
                s['included'] = True
        # populate `paths`
        filter(self)

    def __len__(self):
        return len(self.paths)

    def parallel(self, func, *args, **kwargs):
        """
        Applies a function to all items in `paths` in parallel using
        `joblib.Parallel`.

        You can pass any argument to `joblib.Parallel` by using keyword
        arguments.

        Arguments
        ---------
        func : callable
            the function that will be called; it must accept two arguments
            that are the index of the song and the dataset. Then, it can
            accept all `args` and `kwargs` that are passed to this function:

            >>>  def myfunc(i, dataset, pinco, pal=lino):
            ...     # do not use `filter` and `chunks` here
            ...     print(pinco, pal)
            ...     print(dataset.paths[i])
            ... marco, etto = 4, 5
            ... d = Dataset().filter(datasets='Bach10')
            ... d.parallel(myfunc, marco, n_jobs=8, pal=etto)

            `filter` and `chunks` shouldn't be used.

        Returns
        -------
        list:
            The list of objects returned by each `func`
        """
        joblib_args = [
            k for k, v in inspect.signature(Parallel).parameters.items()
        ]
        joblib_dict = {
            k: kwargs.pop(k)
            for k in dict(kwargs) if k in joblib_args
        }

        return Parallel(**joblib_dict)(delayed(func)(i, self, *args, **kwargs)
                                       for i in tqdm(range(len(self.paths))))

    def get_songs(self):
        """
        Returns a list of dict, each representing a song
        """

        songs = []
        for dataset in self.datasets:
            if dataset['included']:
                for song in dataset['songs']:
                    if song['included']:
                        songs.append(song)
        return songs

    def idx_chunk_to_whole(self, name, idx):
        """
        Given a dataset name and an idx or a list of idx relative to the input
        dataset, returns the idx relative to this whole dataset.

        Use this method if you need, for instance the index of a song for which
        you have the index in a single dataset.
        """
        if type(idx) is int:
            return idx + self._chunks[name][0]
        elif type(idx) is list:
            return [i + self._chunks[name][0] for i in idx]
        else:
            raise Exception('idx should be int or list of int!')

    def get_mix(self, idx, sr=None):
        """
        Returns the audio array of the mixed song

        Arguments
        ---------
        idx : int
            the index of the wanted item
        sr : int or None
            the sampling rate at which the audio will be returned
            (if needed, a resampling is performed). If `None`, no
            resampling is performed

        Returns
        -------
        mix : numpy.ndarray
            the audio waveform of the mixed song
        int :
            The sampling rate of the audio array
        """
        recordings_fn = self.get_mix_paths(idx)

        recordings = []
        for recording_fn in recordings_fn:
            audio, in_sr = utils.open_audio(
                joinpath(self.install_dir, recording_fn))
            recordings.append(audio)

        if len(recordings) > 1:
            L = max(len(rec) for rec in recordings)
            mix = np.zeros(L, dtype=np.float32)
            for rec in recordings:
                mix[:rec.shape[0]] += rec
            mix /= len(rec)
        else:
            mix = recordings[0]

        if sr is not None:
            resampler = Resample(inputSampleRate=in_sr, outputSampleRate=sr)
            mix = resampler(mix)
        else:
            sr = in_sr
        return mix, sr

    def get_gts(self, idx):
        """
        Return the ground-truth of the wanted item

        Arguments
        ---------
        idx : int
            the index of the wanted item

        Returns
        -------
        list :
            list of dictionary representing the ground truth of each single source
        """

        gts = []
        gts_fn = self.get_gts_paths(idx)
        for gt_fn in gts_fn:
            input_fn = joinpath(self.install_dir, gt_fn)

            gt = json.load(gzip.open(input_fn))
            gts.append(gt)
        return gts

    def get_source(self, idx):
        """
        Returns the sources at the specified index

        Arguments
        ---------
        idx : int
            the index of the wanted item

        Returns
        -------
        list :
            a list of numpy.ndarray representing the audio of each source
        int :
            The sampling rate of the audio array
        """
        sources_fn = self.get_sources_paths(idx)

        sources = []
        sr = -1
        for source_fn in sources_fn:
            audio, sr = utils.open_audio(joinpath(self.install_dir, source_fn))
            sources.append(audio)
        return sources, sr

    def get_item(self, idx):
        """
        Returns the mixed audio, sources and ground truths of the specified item.

        Arguments
        ---------
        idx : int
            the index of the wanted item

        Returns
        -------
        numpy.ndarray :
            audio of the mixed sources
        list :
            a list of numpy.ndarray representing the audio of each source
        list :
            list of dictionary representing the ground truth of each single source
        """
        mix = self.get_mix(idx)
        sources = self.get_source(idx)
        gts = self.get_gts(idx)
        return mix, sources, gts

    def get_pianoroll(self,
                      idx,
                      score_type=['misaligned'],
                      resolution=0.25,
                      onsets=False,
                      velocity=True):
        """
        Create pianoroll from list of pitches, onsets and offsets (in this order).

        Arguments
        ---------
        idx : int
            The index of the song to retrieve.
        score_type : list of str
            The key to retrieve the list of notes from the ground_truths. see
            `chose_score_type` for explanation
        resolution : float
            The duration of each column (in seconds)
        onsets : bool
            If True, the value '-1' is put sn each onset
        velocity : bool
            if True, values of each note is the velocity (except the first
            frame if `onsets` is used)

        Returns
        -------
        numpy.ndarray :
            A (128 x n) array where rows represent pitches and columns are time
            instants sampled with resolution provided as argument.

        Note
        ----

        In the midi.org standard, pitches start counting from 0; however,
        sometimes people use to count pitches from 1. Depending on the dataset
        that you are using, verify how pitches are counted. In the ASMD default
        ground-truths, pitches are set with 0-based indexing.

        In case your dataset does not start counting pitches from 0, you should
        correct the output of this function.
        """

        gts = self.get_gts(idx)
        score_type = chose_score_type(score_type, gts)

        # computing the maximum offset
        max_offs = [max(gt[score_type]['offsets']) for gt in gts]
        pianoroll = np.zeros((128, int(max(max_offs) / resolution) + 1))

        # filling pianoroll
        for gt in gts:
            ons = gt[score_type]['onsets']
            offs = gt[score_type]['offsets']
            pitches = gt[score_type]['pitches']
            velocities = gt[score_type]['velocities']
            if not velocities or not velocity:
                velocities = [1] * len(pitches)

            for i in range(len(pitches)):
                p = int(pitches[i])
                on = int(ons[i] / resolution)
                off = int(offs[i] / resolution) + 1

                pianoroll[p, on:off] = velocities[i]
                if onsets:
                    pianoroll[p, on] = -1

        return pianoroll

    def get_beats(self, idx):
        """
        Get a list of beat position in seconds, to be used together with the
        score data.

        Arguments
        ---------
        idx : int
            The index of the song to retrieve.

        Returns
        -------
        numpy.ndarray :
            each row contains beat positions of each ground truth
        """
        gts = self.get_gts(idx)
        beats = []
        for gt in gts:
            beats.append(gt['score']['beats'])

        return np.array(beats)

    def get_score_duration(self, idx):
        """
        Returns the duration of the most aligned score available for a specific
        item
        """
        gts = self.get_gts(idx)
        score_type = chose_score_type(
            ['precise_alignment', 'broad_alignment', 'misaligned', 'score'],
            gts)

        gts_m = 0
        for gt in gts:
            gt_m = max(gt[score_type]['offsets'])
            if gt_m > gts_m:
                gts_m = gt_m
        return gts_m

    def get_audio_data(self, idx):
        """
        Returns audio data of a specific item without loading the full audio.

        N.B. see essentia.standard.MetadataReader!

        Returns
        -------

        list of tuples :
            each tuple is referred to a source and contains the following

        int :
            duration in seconds
        int :
            bitrate (kb/s)
        int :
            sample rate
        int :
            number of channels
        """
        recordings_fn = self.paths[idx][0]

        metadata = []
        for recording_fn in recordings_fn:
            recording_fn = joinpath(self.install_dir, recording_fn)
            reader = MetadataReader(filename=str(recording_fn),
                                    filterMetadata=True)
            metadata.append(reader()[-4:])
        return metadata

    def get_audio(self, idx, sources=None):
        """
        Get the mixed audio of certain sources or of the mix

        Arguments
        ---------
        idx : int
            The index of the song to retrieve.
        sources : list or None
            A list containing the indices of sources to be mixed and returned.
            If `None`, no sources will be mixed and the global mix will be
            returned.

        Returns
        -------
        numpy.ndarray :
            A (n x 1) array which represents the mixed audio.
        int :
            The sampling rate of the audio array
        """

        if sources is not None:
            audio, sr = self.get_source(idx)
            L = max(len(au) for au in audio)
            out = np.zeros(L, dtype=np.float32)
            for i, au in enumerate(audio):
                if i in sources:
                    out[:au.shape[0]] += au
            audio = out / len(audio)
        else:
            audio, sr = self.get_mix(idx)

        return audio, sr

    def get_sources_paths(self, idx) -> List[str]:
        """
        Return paths to single-sources audio recordings, one for each audio

        Returns list of string
        """
        return self.paths[idx][1]

    def get_mix_paths(self, idx) -> List[str]:
        """
        Return paths to the mixed recording if available

        Returns list of string (usually only one)
        """
        return self.paths[idx][0]

    def get_gts_paths(self, idx) -> List[str]:
        """
        Return paths to the ground-truth files, one for each source

        Returns list of string
        """
        return self.paths[idx][2]

    def get_initial_bpm(self, idx) -> Optional[float]:
        """
        Return the initial bpm of the first source if `score` alignment type is
        available at index `idx`, otherwise returns None
        """
        beats = self.get_beats(idx)
        if len(beats[0]) < 2:
            return None

        delta = beats[0][1] - beats[0][0]
        return 60. / delta

    def get_missing_extra_notes(self, idx, kind: str) -> List[np.ndarray]:
        """
        Returns the missing or extra notes of a song. For each source, an array
        of boolean values is returned. If you want the missing/extra notes for
        the whole song, use ``dataset_utils.get_score_mat``

        `kind` can be 'extra' or 'missing'

        """
        gts = self.get_gts(idx)
        out = []
        for gt in gts:
            out.append(np.asarray(gt[kind], dtype=np.bool8))
        return out


def load_definitions(path):
    """
    Given a `path` to a directory, returns a list of dictionaries containing
    the definitions found in that directory (not recursive search)
    """
    datasets = []
    for file in os.listdir(path):
        fullpath = joinpath(path, file)
        if os.path.isfile(fullpath) and fullpath.endswith('.json'):
            # add this dataset
            try:
                print("Opening " + fullpath)
                datasets.append(json.load(open(fullpath, 'rt')))
            except:
                print("Error opening " + fullpath)
    return datasets


def func_wrapper(func, path, *args, **kwargs):
    d = Dataset(empty=True)
    d.paths = [path]
    return func(0, d, *args, **kwargs)
