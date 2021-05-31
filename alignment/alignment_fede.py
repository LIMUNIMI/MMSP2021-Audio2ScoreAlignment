# distutils: language = c

import itertools
import os
from operator import itemgetter
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import plotly.express as px
from dtw import dtw
from dtw.stepPattern import StepPattern
from fastcluster import linkage
from joblib import Parallel, delayed
from julia.api import LibJulia
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

from . import align
from . import settings as s
from .fede_dist import _jaccard_dist, my_intersect
from .fede_dtw import FedeWindow, merge_matching_indices, step_patterns

JuliaMain: Any = None


def include_julia(filename: str):
    api = LibJulia.load()
    api.init_julia(
        ["--project=.", "--optimize=3", f"--threads={os.cpu_count()}"])
    from julia import Main  # noqa: autoimport

    Main.eval(f'include("{filename}")')
    Main.FedeAlignment.precompile("matscore.csv", "matperfm.csv")

    global JuliaMain
    JuliaMain = Main
    return Main


def _chose_sample(arr: np.ndarray, score_reverse: np.ndarray,
                  perfm_reverse: np.ndarray,
                  dist_mat: np.ndarray) -> np.ndarray:
    """
    Given repeated notes, chose the the best one
    """

    to_keep = np.ones(arr.shape[0], dtype=np.bool8)
    unique, counts = np.unique(arr[:, 0], return_counts=True)
    nonunique_val = unique[counts > 1]
    for val in nonunique_val:
        idx = np.where(arr[:, 0] == val)[0]
        clusters1 = score_reverse[val]
        clusters2 = perfm_reverse[arr[idx, 1]]
        # here clusters1 is only 1 number, corresponding to the cluster of the
        # element which is being repeated
        chosen = dist_mat[clusters1, clusters2].argmin()
        to_keep[idx] = False
        to_keep[idx[chosen]] = True
    return to_keep


def _get_unique_path(matched_idx: np.ndarray, score_reverse: np.ndarray,
                     perfm_reverse: np.ndarray,
                     dist_mat: np.ndarray) -> np.ndarray:
    """
    Computes a unique path from a path among notes by removing repeated notes.
    It chooses the couple of notes associated with the two nearest clusters. In
    case of parity, the first couple is taken.
    """
    to_keep = _chose_sample(matched_idx, score_reverse, perfm_reverse,
                            dist_mat)
    matched_idx = matched_idx[to_keep]
    to_keep = _chose_sample(matched_idx[:, ::-1], perfm_reverse, score_reverse,
                            dist_mat.T)
    matched_idx = matched_idx[to_keep]
    return matched_idx


def _compute_dist_mat(clusters1: List[List[int]],
                      clusters2: List[List[int]],
                      dist_func: Callable,
                      th: float,
                      simcarn: bool,
                      win_fn: Callable,
                      win_args: dict = {}) -> np.ndarray:
    """
    Compute distance matrix using `dist_func`
    """

    N = len(clusters1)
    M = len(clusters2)
    out = np.full((N, M), np.inf, dtype=np.float64)

    for i in range(N):
        for j in range(M):

            # limit the computation to radius
            # see slanted band in
            # https://github.com/DynamicTimeWarping/dtw-python/blob/master/dtw/window.py
            if win_fn(i, j, **win_args):
                out[i, j] = dist_func(np.asarray(clusters1[i]),
                                      np.asarray(clusters2[j]), th, simcarn)
    # initial and last note have zero distance from any note
    out[0] = 0
    out[-1] = 0
    out[:, 0] = 0
    out[:, -1] = 0
    return out


def _transform_clusters(
        clusters: np.ndarray,
        mat: np.ndarray) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    Takes a list of cluster labels (as usually returned by scipy) and transform
    it in a list of lists of indices/features; Also returns a list of lables
    where labels are refered to the transfermed index of each cluster; this
    last list can be used to reverse the representation of clusters.
    """
    # creating the list of clusters
    features: List[List[int]] = []
    transformed_clusters: List[List[int]] = []
    reverse: List[int] = []
    this_cluster = -1
    counter = -1
    for i in range(clusters.shape[0]):
        if clusters[i] != this_cluster:
            this_cluster = clusters[i]
            counter += 1
            features.append([])
            transformed_clusters.append([])
        features[counter].append(int(mat[i, 0]))
        transformed_clusters[counter].append(i)
        reverse.append(counter)
    return features, transformed_clusters, reverse


def _matching_notes_clusters(score_clusters, score_features, perfm_clusters,
                             perfm_features, th: float,
                             simcarn: bool) -> np.ndarray:
    """
    Here, a cluster is a list of pitch. We take the list of clusters and
    compare each note inside each cluster to match the corresponding notes. The
    match is done based on the features (pitch), but the indices of the
    corresponding notes are returned; this is why we need both the list of
    cluster features and the list of cluster note indices.
    """
    out = []
    for i in range(len(score_clusters)):
        _p1, _p2, matched_idx1, matched_idx2 = my_intersect(
            score_features[i],
            perfm_features[i],
            th,
            return_indices=True,
            simcarn=simcarn,
            broadcast=True)
        if len(matched_idx1) > 0:
            out.append(
                np.stack([
                    np.array(score_clusters[i])[matched_idx1],
                    np.array(perfm_clusters[i])[matched_idx2]
                ],
                         axis=1))

    return np.concatenate(out)


def _clusterize(
    mat: np.ndarray,
    threshold: Optional[float] = None,
    num_clusters: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    Returns clusters of notes by onsets using single-linkage and stopping
    agglomeration procedure when `threshold` is reached.

    Returns the features of each cluster, ordered by onsets, the index of notes
    in each cluster, ordered by onset, and the list of cluster of each note.
    Features is only one (pitch).

    If `num_clusters` is not None, it should be a int and `maxclust` criterion
    is used in place of the threshold (which is not used)
    """
    # clusterize the onsets
    p = pdist(mat[:, 1, None], metric='minkowski', p=1.)
    Z = linkage(p, method='single')
    if num_clusters is None:
        clusters = fcluster(Z, t=threshold, criterion='distance')
    else:
        clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

    # creating the list of clusters
    features, clusters, reverse = _transform_clusters(clusters, mat)

    return features, clusters, reverse


def plot_path(dist_mat, path):

    fig = px.imshow(dist_mat)
    for y, x in path:
        fig.add_annotation(text='x',
                           x=x,
                           xref="x",
                           y=y,
                           yref="y",
                           showarrow=False,
                           font_color="white",
                           font_size=15)
    fig.show()


def fede_align(matscore,
               matperfm,
               score_th: Optional[float] = 0.0,
               plot=s.PLOT,
               thresholds=[1, s.FEDE_PITCH_TH],
               dist: Callable = s.FEDE_DIST,
               audio_data: Optional[Tuple[np.ndarray, int]] = None):
    """
    Returns new onsets and offsets
    """

    matched_idx, _matscore, _matperfm = get_matching_notes(
        matscore,
        matperfm,
        score_th=score_th,
        dist=dist,
        plot=plot,
        thresholds=thresholds)

    # perform the alignment
    res = align.align_with_matching_notes(matched_idx, _matscore, _matperfm,
                                          audio_data, plot)
    if res is None:
        return None
    else:
        return res[0], res[1]


def get_matching_notes(matscore,
                       matperfm,
                       score_th: Optional[float] = None,
                       dist=s.FEDE_DIST,
                       plot=s.PLOT,
                       step_patterns=step_patterns,
                       thresholds=[1.0, s.FEDE_PITCH_TH]):
    """
    Reapeatly calls ``_get_matching_notes`` and merge the indices.

    Returns a mapping of indices between notes in `matscore` and in `matperfm`
    with shape (N, 2), where N is the number of matching notes.
    Also returns `matscore` and `matperfm` with initial and ending virtual
    notes: you can safely discard them by slicing with `[1:-1]`, but returned
    indices are referred to the returned mats.

    If `score_th` is None, then the number of clusters for the score is
    inferred from the performance (whose threshold is fixed in `settings.py`)

    `thresholds` is a list of thresholds used; a value of 1 causes simcarn not
    being used, a value != 1 causes simcarn to be used

    `dist` is the function used for thresholds != 1 (using simcarn)
    """

    if s.BACKEND == 'julia':
        global JuliaMain
        if JuliaMain is None:
            include_julia("alignment/alignment_fede.jl")

        matched_idx, _matscore, _matperfm =\
            JuliaMain.FedeAlignment.get_matching_notes(
                matscore, matperfm, s.ALPHA, s.BETA, s.SP_WEIGHT, score_th,
                thresholds)
        # fixing julia indices...
        matched_idx -= 1
    else:
        configs = list(itertools.product(step_patterns, thresholds))
        matching = Parallel(
            n_jobs=len(configs) if s.PARALLEL else 1,
            backend='multiprocessing')(delayed(_get_matching_notes)(
                matscore,
                matperfm,
                _jaccard_dist if th == 1 else dist,
                step_pattern,
                th,
                s.ALPHA,
                s.BETA,
                score_th,
                plot,
                simcarn=th != 1,
                index=i) for i, (step_pattern, th) in enumerate(configs))
        _matscore = matching[0][1]
        _matperfm = matching[0][2]
        matched_indices = list(zip(*matching))[0]

        # merging matching indices
        matched_idx = merge_matching_indices(matched_indices)

    # print(len(matched_idx))
    return matched_idx, _matscore, _matperfm


def print_data(step, th):
    print(step)
    print("th:", th)


def _get_matching_notes(
        matscore: np.ndarray,
        matperfm: np.ndarray,
        dist: Callable,
        step_pattern: StepPattern,
        th: float,
        alpha: int,
        beta: int,
        score_th: Optional[float] = 0.0,
        plot: bool = False,
        simcarn: bool = False,
        index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a mapping of indices between notes in `matscore` and in `matperfm`
    with shape (N, 2), where N is the number of matching notes.
    Also returns `matscore` and `matperfm` with initial and ending virtual
    notes: you can safely discard them by slicing with `[1:-1]`, but returned
    indices are referred to the returned mats.

    If `score_th` is None, then the number of clusters for the score is
    inferred from the performance (whose threshold is fixed in `settings.py`)
    """
    # FedeAlignment = JuliaMain.FedeAlignment
    # FedeDist = FedeAlignment.FedeDist
    # FedeDTW = FedeAlignment.FedeDTW
    # DTW = FedeAlignment.DTW
    # print_data(step_pattern, th)

    # inserting starting and ending notes
    start = -1
    end = max(matscore[:, 2].max(), matperfm[:, 2].max()) + 1
    starting = np.full((1, matscore.shape[1]), start, dtype=matscore.dtype)
    ending = np.full((1, matscore.shape[1]), end, dtype=matscore.dtype)
    matscore = np.concatenate([starting, matscore, ending])
    matperfm = np.concatenate([starting, matperfm, ending])

    # compute clusters
    perfm_features, perfm_clusters, perfm_reverse = _clusterize(
        matperfm, s.CLUSTER_TH)

    # jperfm_features, jperfm_clusters, jperfm_reverse = FedeAlignment._clusterize(
    #     matperfm, threshold=0.05)
    # if np.any(jperfm_reverse - 1 != perfm_reverse):
    #     print("Error for this data at clustering perfm:")
    #     print_data(step_pattern, th)
    #     __import__('ipdb').set_trace()

    if score_th is not None:
        score_features, score_clusters, score_reverse = _clusterize(
            matscore, threshold=score_th)
        # jscore_features, jscore_clusters, jscore_reverse = FedeAlignment._clusterize(
        #     matscore, threshold=score_th)
    else:
        score_features, score_clusters, score_reverse = _clusterize(
            matscore, num_clusters=len(perfm_clusters))
        # jscore_features, jscore_clusters, jscore_reverse = FedeAlignment._clusterize(
        #     matscore, num_clusters=len(perfm_clusters))
    # if np.any(jscore_reverse - 1 != score_reverse):
    #     print("Error for this data at clustering score:")
    #     print_data(step_pattern, th)
    #     __import__('ipdb').set_trace()

    window_fn = FedeWindow(score_features,
                           perfm_features,
                           dist,
                           alpha=alpha,
                           beta=beta,
                           min_radius=5,
                           dist_args=dict(th=th, simcarn=simcarn))
    # jdist_fn = FedeDist.JaccardDist(th, simcarn)
    # jwindow_fn = FedeDTW.get_fede_win(
    #     FedeDTW.FedeWindowParam(jdist_fn, 5, alpha, beta), jscore_features,
    #     jperfm_features)
    # if np.any(jwindow_fn.mask != window_fn.mask):
    #     print("Error while computing window mask")
    #     print_data(step_pattern, th)
    #     __import__('ipdb').set_trace()

    dist_mat = _compute_dist_mat(score_features, perfm_features, dist, th,
                                 simcarn, window_fn)
    # jdist_mat = FedeAlignment._compute_dist_mat(jscore_features,
    #                                             jperfm_features, jdist_fn,
    #                                             jwindow_fn)
    # if np.any(dist_mat != jdist_mat):
    #     print("Error while computing dist_mat")
    #     print_data(step_pattern, th)
    #     __import__('ipdb').set_trace()

    result = dtw(x=dist_mat, step_pattern=step_pattern, window_type=window_fn)
    # result = dtw(x=dist_mat,
    #              step_pattern=step_pattern,
    #              window_type=window_fn,
    #              keep_internals=True)
    # find unique path
    path = np.stack([result.index1, result.index2], axis=1)

    # jstep_pattern = DTW.step_patterns[index // 2]
    # jmatrix = DTW.dtw_matrix(jdist_mat, window_fn=jwindow_fn, step_pattern=jstep_pattern)
    # mt = result.costMatrix
    # mt[np.isnan(mt)] = np.inf
    # if np.any(jmatrix != mt):
    #     print("Error while computing dtw")
    #     print_data(step_pattern, th)
    #     __import__('ipdb').set_trace()

    # jpath = DTW.trackback(jmatrix, jstep_pattern)
    # jpath = jpath[np.lexsort((jpath[:, 1], jpath[:, 0])), :]
    # mt = path[np.lexsort((path[:, 1], path[:, 0])), :]
    # if np.any(jpath - 1 != mt):
    #     print("Error while computing path")
    #     print_data(step_pattern, th)
    #     __import__('ipdb').set_trace()

    if plot:
        print("---")
        print(f"Using {step_pattern}:")
        print(result.stepPattern)
        plot_path(dist_mat, path)

    # computing the matched notes
    matched_idx = _matching_notes_clusters(
        itemgetter(*path[:, 0])(score_clusters),
        itemgetter(*path[:, 0])(score_features),
        itemgetter(*path[:, 1])(perfm_clusters),
        itemgetter(*path[:, 1])(perfm_features), th, simcarn)

    # compute unique matched notes
    matched_idx = _get_unique_path(matched_idx, np.asarray(score_reverse),
                                   np.asarray(perfm_reverse), dist_mat)
    return matched_idx, matscore, matperfm
