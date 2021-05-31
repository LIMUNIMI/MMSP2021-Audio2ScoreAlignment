import os
import signal
import sys
import uuid
from pathlib import Path
from subprocess import Popen, TimeoutExpired
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import align
from . import settings as s
from . import utils


def which(program: Union[str, Path]):
    """
    Stolen from stackoverflow
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def check_executables():
    """
    1. check if program compiled exists
    2. if not, suggests the command line to compile it

    Returns True if exists, False otherwise
    """

    for ex in [
            'midi2pianoroll', 'SprToFmt3x', 'Fmt3xToHmm', 'ScorePerfmMatcher',
            'ErrorDetection', 'RealignmentMOHMM', 'MetchToCorresp'
    ]:
        if not which(s.EITA_PATH / 'Programs' / ex):
            print(
                "Eita tools seems to be uncorrectly compiled, please use the following command to compile",
                file=sys.stderr)
            print(f"`{s.EITA_PATH}/compile.sh`")


def remove_temp_files(path: Union[str, Path]):
    path = Path(path)
    for file in path.parent.glob(path.stem + "*"):
        file.unlink()


def eita_align(
    matscore: np.ndarray,
    matperfm: np.ndarray,
    audio_data: Optional[Tuple[np.ndarray, int]] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    0. runs the alignment
    2. clean the output files
    3. returns new onsets and offsets or None if something fails
    """
    matched_idx = get_matching_notes(matscore, matperfm)
    if matched_idx is None:
        return None

    res = align.align_with_matching_notes(matched_idx, matscore, matperfm,
                                          audio_data)
    if res is None:
        return None
    else:
        return res[0], res[1]


def get_matching_notes(matscore: np.ndarray,
                       matperfm: np.ndarray,
                       timeout=None):
    """
    Returns a mapping of indices between notes in `matscore` and in `matperfm`
    with shape (N, 2), where N is the number of matching notes.

    Performs Eita Nakamura alignment in a separate process and waits for it.
    This cleans all the output files. Returns None if the separate process
    fails.
    """
    p1 = uuid.uuid1(clock_seq=os.getpid())
    p2 = uuid.uuid1(clock_seq=os.getpid())
    path1 = str(p1) + '.mid'
    path2 = str(p2) + '.mid'

    # writing music data to midi files
    # the first argument is the reference signal
    utils.mat2midi(matperfm, path1)
    utils.mat2midi(matscore, path2)

    try:
        # Launch the external process
        popen = Popen(
            [f"{s.EITA_PATH}/MIDIToMIDIAlign.sh", path1[:-4], path2[:-4]])

        popen.wait(timeout=timeout)

        eita_data = pd.read_csv(path2[:-4] + "_corresp.txt",
                                sep="\t",
                                skiprows=1,
                                header=None,
                                index_col=False)
    except FileNotFoundError:
        return None
    finally:
        # remove files created by eita code
        remove_temp_files(path1)
        remove_temp_files(path2)

    # removing rows with -1 (not matched notes)
    eita_data.columns = [
        "alignID", "alignOntime", "alignSitch", "alignPitch", "alignOnvel",
        "refID", "refOntime", "refSitch", "refPitch", "refOnvel", "empty"
    ]
    eita_data = eita_data[eita_data["alignOntime"] != -1]
    eita_data = eita_data[eita_data["refOntime"] != -1]

    return np.stack(
        [eita_data['alignID'].astype(int), eita_data['refID'].astype(int)],
        axis=1)
