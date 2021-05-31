#!/usr/bin/env python3

from typing import Callable, Optional, Tuple

import numpy as np
import plotly.express as px

from . import settings as s
from . import transcription
from .seba.align import audio_to_score_alignment


def collapse_repeating_values(mapping: np.ndarray) -> np.ndarray:
    """
    For each repeating value in `mapping[0]`, add only one mapping which maps
    to the average of the corresponging `mapping[1]`. This also sorts the
    returned mapping so that it's ready for interpolation
    """

    unique = np.unique(mapping[0])
    out = np.zeros((2, unique.shape[0]), dtype=np.float64)
    for i in range(unique.shape[0]):
        idx = np.where(mapping[0] == unique[i])[0]
        val = mapping[1, idx]
        if idx.shape[0] > 1:
            val = val.mean()

        out[:, i] = (unique[i], val)

    return out[:, out[0].argsort()]


def align_with_matching_notes(matched_idx: np.ndarray,
                              matscore: np.ndarray,
                              matperfm: np.ndarray,
                              audio_data: Optional[Tuple[np.ndarray,
                                                         int]] = None,
                              plot: bool = s.PLOT):
    """
    Given a mapping of matching notes and two scores, copies matching times
    from `matperfm` to `matscore` and infers the remaining note positions

    Works in-place modifying matscore

    If `audio_data` is provided, seba method with FastDTW is used to realign
    non-matched notes.
    """
    # computing the onset mapping
    ons_time_map = np.stack(
        [matscore[matched_idx[:, 0], 1], matperfm[matched_idx[:, 1], 1]])
    ons_time_map = collapse_repeating_values(ons_time_map)

    # computing the duration mapping
    dur_matscore = matscore[matched_idx[:, 0]]
    dur_matscore = dur_matscore[:, 2] - dur_matscore[:, 1] + 1e-15
    dur_matperfm = matperfm[matched_idx[:, 1]]
    dur_matperfm = dur_matperfm[:, 2] - dur_matperfm[:, 1] + 1e-15
    dur_time_map = np.stack(
        [matscore[matched_idx[:, 0], 1], dur_matperfm / dur_matscore])
    dur_time_map = collapse_repeating_values(dur_time_map)

    if plot:
        fig = px.line(x=ons_time_map[0],
                      y=ons_time_map[1],
                      title='ons mapping')
        fig.update_traces(mode='markers+lines')
        fig.show()
        fig = px.line(x=dur_time_map[0],
                      y=dur_time_map[1],
                      title='dur mapping')
        fig.update_traces(mode='markers+lines')
        fig.show()

    # copying the common notes
    matscore[matched_idx[:, 0], 1:3] = matperfm[matched_idx[:, 1], 1:3]

    # looking for missing notes
    missing_idx = np.setdiff1d(np.arange(matscore.shape[0]),
                               matched_idx[:, 0],
                               assume_unique=True)
    # looking for duration in the score of missing notes
    missing_matscore_dur = matscore[missing_idx, 2] - matscore[missing_idx, 1]

    # interpolating onsets
    matscore[missing_idx, 1] = np.interp(matscore[missing_idx, 1],
                                         ons_time_map[0], ons_time_map[1])

    # interpolating duration
    matscore[missing_idx, 2] = np.interp(matscore[missing_idx, 1],
                                         dur_time_map[0], dur_time_map[1])
    matscore[missing_idx, 2] *= missing_matscore_dur
    matscore[missing_idx, 2] += matscore[missing_idx, 1]

    missing_idx = np.sort(missing_idx)

    if audio_data is not None:
        # running seba method to adjust inferred notes
        res = audio_to_score_alignment(matscore.copy(),
                                       audio_data,
                                       fastdtw=True)
        if res is None:
            return matscore[:, 1], matscore[:, 2]
        else:
            matscore[:, 1] = res[0]
            matscore[:, 2] = res[1]

            # copying the common notes again (DTW has modified them)
            matscore[matched_idx[:, 0], 1:3] = matperfm[matched_idx[:, 1], 1:3]
    return matscore[:, 1], matscore[:, 2]


def align_to_midi(matscore: np.ndarray, matperfm: np.ndarray, method: Callable,
                  *args, **kwargs) -> np.ndarray:
    """
    Works in-place modifying matscore and returning it. Also removes all
    pedaling and velocities.
    """

    # aligning
    res = method(matscore, matperfm, *args, **kwargs)
    if res is not None:
        new_ons, new_offs = res
    else:
        raise RuntimeError(f"{method} output is None")
    matscore[:, 1] = new_ons
    matscore[:, 2] = new_offs
    matscore[:, 3] = 64
    return matscore


def align_to_audio(matscore: np.ndarray, audio_data: Tuple[np.ndarray, int],
                   transcr_method: str, align_method: Callable, *args,
                   **kwargs) -> np.ndarray:
    """
    Works in-place modifying matscore and returning it. Also removes all
    pedaling and velocities.
    """
    # mattrans = transcription.transcribe(audio_data[0], audio_data[1],
    #                                     transcr_method)
    mattrans = transcription.transcribe(audio_data[0], audio_data[1],
                                        transcr_method)
    matscore = align_to_midi(matscore, mattrans, align_method)

    return matscore
