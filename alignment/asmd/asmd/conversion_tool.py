import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import tarfile
from copy import deepcopy
from difflib import SequenceMatcher
from os.path import join as joinpath
from typing import Callable, List, Optional

import numpy as np
from pretty_midi.constants import INSTRUMENT_MAP

from .asmd import load_definitions
from .convert_from_file import *
from .convert_from_file import _sort_alignment, _sort_pedal
from .idiot import THISDIR

# this is only for detecting the package path

#: if True, run conversion in parallel processes
PARALLEL = True
# PARALLEL = False

rng = np.random.default_rng(1002)


def normalize_text(text):
    return ''.join(ch for ch in text if ch.isalnum()).lower()


def text_similarity(a, b):
    return SequenceMatcher(None, a,
                           b).find_longest_match(0, len(a), 0, len(b)).size


# normalizing MIDI instrument names
INSTRUMENT_MAP = list(map(normalize_text, INSTRUMENT_MAP))

INSTRUMENT_MAP.append('drumkit')


def _is_sorted(a_list: list) -> bool:
    """
    Return True if `a_list` is sorted, False otherwise
    """
    for i in range(len(a_list) - 1):
        if a_list[i] > a_list[i + 1]:
            return False
    return True


def _is_in_range(a_list: list,
                 min: Optional[float] = None,
                 max: Optional[float] = None) -> bool:
    """
    Return True if `a_list` ontains values between `min` and `max`, False
    otherwise
    """
    for el in a_list:
        if min is not None:
            if el < min:
                return False
        if max is not None:
            if el > max:
                return False
    return True


def _is_greater(list1: list, list2: list):
    """
    return True if `list1[i] > list2[i]` for each `i`
    """
    return all([list1[i] > list2[i] for i in range(len(list1))])


def check(gt: dict) -> int:
    """
    Returns 0 if the ground_truth dictionary representation is correctly built,
    a value > 0 otherwise
    """

    for alignment in [
            'precise_alignment', 'broad_alignment', 'score', 'misaligned'
    ]:
        # check that onsets are sorted
        if not _is_sorted(gt[alignment]['onsets']):
            return 1
        # check that onsets are > 0
        if not _is_in_range(gt[alignment]['onsets'], 0, None):
            return 2
        # check that offsets are > 0
        if not _is_in_range(gt[alignment]['offsets'], 0, None):
            return 3
        # check that offsets are > onsets
        if not _is_greater(gt[alignment]['offsets'], gt[alignment]['onsets']):
            return 4
        # check that pitches are in 0-127
        if not _is_in_range(gt[alignment]['pitches'], 0, 127):
            return 5
        # check that velocities are in 0-127
        if not _is_in_range(gt[alignment]['velocities'], 0, 127):
            return 6

    # check that ['score']['beats'] is sorted
    if not _is_sorted(gt['score']['beats']):
        return 7
    # check length of 'missing' and 'extra'
    if len(gt['missing']) != len(gt['extra']):
        return 8
    # check that 'f0' is > 0
    if not _is_in_range(gt['f0'], 0, None):
        return 9

    for pedal in ['soft', 'sostenuto', 'sustain']:
        # check that 'times' is sorted
        if not _is_sorted(gt[pedal]['times']):
            return 10
        # check that 'values' are in 0-127
        if not _is_in_range(gt[pedal]['values'], 0, 127):
            return 11

    return 0


###################################################
def slice_intersection(a: slice, b: slice):
    if b.start < a.start < b.stop or a.start < b.start < a.stop:
        return False
    else:
        return True


def random_subslice(length: int, slice_: slice):
    start = rng.integers(slice_.start, slice_.stop - length)
    stop = start + length
    return slice(start, stop)


def random_distinct_subslices(tot: int, size: int) -> List[slice]:
    slice_ = slice(0, size)
    slices: List[slice] = []
    nnn = 0
    while nnn < tot:
        while True:
            new_slice = random_subslice(rng.integers(1, tot - nnn + 1), slice_)
            if all(slice_intersection(new_slice, r) for r in slices):
                nnn += new_slice.stop - new_slice.start
                slices.append(new_slice)
                break
    slices.sort(key=lambda r: r.start)
    return slices


###################################################


def merge_dicts(idx, *args):
    """
    Merges lists of dictionaries, by adding each other the values of
    corresponding dictionaries

    `args` can contain `None` values (except for the first one); in such case,
    that argument will be skipped; this is useful if a dataset contains
    annotations not for all the files (e.g. ASAP annotations for Maestro
    dataset)
    """

    assert all(type(x) is list or x is None
               for x in args), "Input types must be lists or None"

    assert all(len(x) == len(args[0]) for x in args[1:]
               if x is not None), "Cannot merge list with different lenghts"

    idx = min(idx, len(args[0]) - 1)  # For PHENICX

    if len(args) == 1:
        return args[0][idx]

    obj1_copy = deepcopy(args[0][idx])

    for arg in args[1:]:
        if arg is not None:
            arg = arg[idx]
            for key in obj1_copy.keys():
                d1_element = obj1_copy[key]
                if type(d1_element) is dict:
                    obj1_copy[key] = merge_dicts(0, [d1_element], [arg[key]])
                elif type(d1_element) is int:
                    obj1_copy[key] = min(d1_element, arg[key])
                else:
                    obj1_copy[key] = d1_element + arg[key]
        # del arg

    return obj1_copy


def fix_offsets(onsets, offsets, pitches):
    """
    Make each offset smaller than the following onset in the
    same pitch.

    Modify offsets in-place.
    """

    # a table to search for same pitches
    table_pitches = [[]] * 128
    for i, p in enumerate(pitches):
        table_pitches[int(p)].append(i)

    for i in range(len(onsets)):
        # search next note with same pitch
        j = None
        for k in table_pitches[int(pitches[i])]:
            if onsets[k] > onsets[i]:
                j = k
                break

        if j is not None and j < len(onsets):
            if offsets[i] > onsets[j]:
                offsets[i] = onsets[j] - 0.005
                if offsets[i] < onsets[i]:
                    offsets[i] = 2 * (onsets[i] + onsets[j]) / 3

        # minimum duration in 0.0625
        offsets[i] = max(onsets[i] + 0.0625, offsets[i])


def misalign(out, stats):
    """
    Given a ground truth dictionary and a `alignment_stats.Stats` object,
    computes onsets and offsets misaligned. Return 3 lists (pitches, onsets,
    offsets).
    """
    if len(out['precise_alignment']['onsets']) > 0:
        alignment = 'precise_alignment'
    else:
        alignment = 'broad_alignment'
    onsets = stats.get_random_onsets(out[alignment]['onsets'])
    offsets = stats.get_random_offsets(out[alignment]['onsets'],
                                       out[alignment]['offsets'], onsets)
    pitches = out[alignment]['pitches']

    fix_offsets(onsets, offsets, pitches)

    return pitches, onsets.tolist(), offsets.tolist()


def conversion(arg):
    """
    A function that is run on each song to convert its ground_truth.
    Intended to be run in parallel.
    """
    l, song, json_file, dataset, stats = arg
    print(" elaborating " + song['title'])
    paths = song['ground_truth']

    to_be_included_in_the_archive = []

    for i, path in enumerate(paths):
        final_path = os.path.join(json_file['install_dir'], path)
        # get the index of the track from the path
        idx = path[path.rfind('-') + 1:path.rfind('.json.gz')]

        # calling each function listed in the map and merge everything

        out = merge_dicts(
            int(idx), *[
                eval(func)(final_path, **params)
                for func, params in dataset["install"]["conversion"]
            ])
        _sort_alignment('score', out)
        _sort_alignment('broad_alignment', out)
        _sort_alignment('precise_alignment', out)
        _sort_pedal(out)

        # take the General Midi program number associated with the most
        # similar instrument name
        instrument = normalize_text(song['instruments'][i])
        out['instrument'] = max(
            range(len(INSTRUMENT_MAP)),
            key=lambda x: text_similarity(INSTRUMENT_MAP[x], instrument))

        # check if at least one group to which this song belongs to has
        # `misaligned` set to 2
        misaligned = False
        for group in song['groups']:
            dataset['ground_truth'][group]['misaligned'] == 2
            misaligned = True
            break

        if misaligned and stats:
            # computing deviations for each pitch
            stats.new_song()
            pitches, onsets, offsets = misalign(out, stats)
            out['misaligned']['onsets'] = onsets
            out['misaligned']['offsets'] = offsets
            out['misaligned']['pitches'] = pitches
            # computing the percentage of missing and extra notes (between 0.10
            # and 0.30)
            L = len(pitches)
            missing, extra = generate_missing_extra(L)
            out['missing'] = missing.tolist()
            out['extra'] = extra.tolist()

        print("   saving " + final_path)
        # pretty printing stolen from official docs
        json.dump(out, gzip.open(final_path, 'wt'), sort_keys=True, indent=4)

        # starting debugger if something is wrong
        if check(out) > 0:
            if PARALLEL:
                print("To start the debugger turn `PARALLEL` to False")
                sys.exit(1)
            else:
                print(
                    "Error: a ground-truth has not passed the checks, starting debugger!"
                )
                __import__('ipdb').set_trace()

        to_be_included_in_the_archive.append(final_path)
    return to_be_included_in_the_archive


def generate_missing_extra(L, min_perc=0.10, max_perc=0.50):
    tot = rng.integers(min_perc * L, max_perc * L)
    me_proportion = rng.random() / 2 + 0.25
    slices = random_distinct_subslices(tot, L)
    missing = np.zeros(L, dtype=np.bool8)
    extra = np.zeros(L, dtype=np.bool8)
    for region in slices:
        if rng.random() > me_proportion:
            missing[region] = True
        else:
            extra[region] = True
    return missing, extra


def create_gt(data_fn,
              gztar=False,
              alignment_stats=None,
              whitelist=[],
              blacklist=[]):
    """
    Parse the json file `data_fn` and convert all ground_truth to our
    representation. Then dump it according to the specified paths. Finally,
    if `gztar` is True, create a gztar archive called 'ground_truth.tar.gz' in
    this directory containing only the ground truth file in their final
    positions.

    If ``alignment_stats`` is not None, it should be an object of type
    ``alignment_stats.Stats`` as the one returned by
    ``alignment_stats.get_stats``
    """

    print("Opening JSON file: " + data_fn)

    json_file = json.load(open(data_fn, 'r'))

    to_be_included_in_the_archive = []
    datasets = load_definitions(joinpath(THISDIR, 'definitions'))
    for dataset in datasets:
        if blacklist:
            if dataset['name'] in blacklist:
                print(dataset['name'] + " in blacklist!")
                continue

        if whitelist:
            if dataset['name'] not in whitelist:
                print(dataset['name'] + " not in whitelist!")
                continue

        if not os.path.exists(
                os.path.join(json_file["install_dir"], dataset["name"])):
            print(dataset["name"] + " not installed, skipping it")
            continue

        print("\n------------------------\n")
        print("Starting processing " + dataset['name'])
        arg = [(i, song, json_file, dataset, alignment_stats)
               for i, song in enumerate(dataset['songs'])]
        if not PARALLEL:
            for i in range(len(dataset['songs'])):
                to_be_included_in_the_archive += conversion(arg[i])
        else:
            CPU = os.cpu_count() - 1  # type: ignore
            p = mp.Pool(CPU)
            result = p.map_async(conversion, arg,
                                 len(dataset['songs']) // CPU + 1)
            to_be_included_in_the_archive += sum(result.get(), [])

    def _remove_basedir(x):
        x.name = x.name.replace(json_file['install_dir'][1:] + '/', '')
        return x

    # creating the archive
    if gztar:
        print("\n\nCreating the final archive")
        with tarfile.open('ground_truth.tar.gz', mode='w:gz') as tf:
            for fname in to_be_included_in_the_archive:
                # adding file with relative path
                tf.add(fname, filter=_remove_basedir)
