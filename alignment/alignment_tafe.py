import numpy as np
# from dtw import dtw
import fastdtw

from . import utils, cdist

# from scipy.spatial.distance import cosine

START_NOTE = 21
EPS = np.finfo(np.float64).eps
#: how many realignment do
NUM_REALIGNMENT = 3
#: how many seconds for each hop size in fine alignment
FINE_HOP = [5, 2.5, 0.5]
# FINE_HOP = [90 / (2**i) for i in range(NUM_REALIGNMENT)]
#: how many seconds for each window in fine alignment
FINE_WIN = [10, 5, 1]


def _my_prep_inputs(x, y, dist):
    """
    Fastdtw sucks too and convicts you to use float64...
    """
    return x, y


def dtw_align(pianoroll, audio_features, misaligned, res: float, radius: int,
              # dist: str, step: str):
              dist: str):
    """
    perform alignment and return new times
    """

    # parameters for dtw were chosen with midi2midi on musicnet (see dtw_tuning)
    # hack to let fastdtw accept float32
    fastdtw._fastdtw.__prep_inputs = _my_prep_inputs
    _D, path = fastdtw.fastdtw(pianoroll.astype(np.float32).T,
                               audio_features.astype(np.float32).T,
                               dist=getattr(cdist, dist),
                               radius=radius)

    # result = dtw(x=cdist.cdist(pianoroll.T, audio_features.T,
    #                      metric=dist).astype(np.float64),
    # result = dtw(x=pianoroll.T, y=audio_features.T,
    #              dist_method=dist,
    #              step_pattern=step,
    #              window_type='slantedband',
    #              window_args=dict(window_size=radius))
    # path = np.stack([result.index1, result.index2], axis=1)

    path = np.array(path) * res
    new_ons = np.interp(misaligned[:, 1], path[:, 0], path[:, 1])
    new_offs = np.interp(misaligned[:, 2], path[:, 0], path[:, 1])

    return new_ons, new_offs


def get_usable_features(matscore, matperfm, res):
    """
    compute pianoroll and remove extra columns
    """
    utils.mat_prestretch(matscore, matperfm)
    score_pr = utils.make_pianoroll(
        matscore, res=res, velocities=False) + utils.make_pianoroll(
            matscore, res=res, velocities=False, only_onsets=True)
    perfm_pr = utils.make_pianoroll(
        matperfm, res=res, velocities=False) + utils.make_pianoroll(
            matperfm, res=res, velocities=False, only_onsets=True)

    return score_pr, perfm_pr


def tafe_align(matscore, matperfm, res=0.02, radius=178, dist='cosine',
               # step='symmetric2'):
               ):
    """
    Returns new onsets and offsets

    Works in-place modifying matscore
    """

    score_pr, perfm_pr = get_usable_features(matscore, matperfm, res)
    # first alignment
    new_ons, new_offs = dtw_align(score_pr, perfm_pr, matscore, res, radius,
                                  # dist, step)
                                  dist)
    matscore[:, 1] = new_ons
    matscore[:, 2] = new_offs

    #     # realign segment by segment
    #     for j in range(NUM_REALIGNMENT):
    #         score_pr, perfm_pr = get_usable_features(matscore, matperfm, res)
    #         hop_size = int(FINE_HOP[j] // res)
    #         win_size = int(FINE_WIN[j] // res)
    #         num_win = int(score_pr.shape[1] // hop_size)
    #         for i in range(num_win):
    #             start = i * hop_size
    #             end = min(i * hop_size + win_size, score_pr.shape[1])
    #             indices_of_notes_in_win = np.argwhere(
    #                 np.logical_and(matscore[:, 1] >= start * res,
    #                                matscore[:, 2] <= end * res))
    #             if indices_of_notes_in_win.shape[0] > 1:
    #                 indices_of_notes_in_win = indices_of_notes_in_win[0]
    #             else:
    #                 continue
    #             score_win = score_pr[:, start:end]
    #             perfm_win = perfm_pr[:, start:end]
    #             ons_win, offs_win = dtw_align(score_win,
    #                                           perfm_win,
    #                                           matscore[indices_of_notes_in_win],
    #                                           res,
    #                                           radius=1,
    #                                           dist=dist)
    #             matscore[indices_of_notes_in_win, 1] = ons_win
    #             matscore[indices_of_notes_in_win, 2] = offs_win

    return matscore[:, 1], matscore[:, 2]
