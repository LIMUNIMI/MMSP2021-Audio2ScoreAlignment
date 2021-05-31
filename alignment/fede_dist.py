from typing import Callable, List

import numpy as np


def simcarn_diff(diff):
    diff = diff % 12
    if diff == 0:
        return 0
    elif diff == 1:
        # minor second
        return 0.8
    elif diff == 2:
        # major second
        return 0.8
    elif diff == 3:
        # minor third
        return 0.25
    elif diff == 4:
        # major third
        return 0.25
    elif diff == 5:
        # perfect fourth
        return 0.7
    elif diff == 6:
        # diminished fifth
        return 0.5
    elif diff == 7:
        # perfect fifth
        return 0.25
    elif diff == 8:
        # minor sixth
        return 0.6
    elif diff == 9:
        # major sixth
        return 0.6
    elif diff == 10:
        # minor seventh
        return 0.5
    elif diff == 11:
        # major seventh
        return 0.8
    else:
        return 1


simcarn_diff = np.vectorize(simcarn_diff, otypes=[np.float64])


def my_intersect(x,
                 y,
                 th: float,
                 broadcast: bool = True,
                 return_count: bool = False,
                 return_indices: bool = False,
                 simcarn: bool = True):
    """
    In this doc, `f` is for `my_intersect`

    An intersection function which also matches elements whose L1 difference is
    `< th`. If `simcarn` is True, The Simonetta-Carnovalini distance between
    pitches is used and the threshold should change accordingly.

    If `return_count` is False, this function returns the matching elements for
    `x` and `y` (2 tuples).

    If `return_count` is True, this function returns the number of matching
    elements for both `x` and `y`. These two numbers can be different.

    If `broadcast` is True, elements are broadcasted so that the number of
    returned elements is the same for `x` and `y`.

    If `return_indices` is True, 2 additional arrays are returned, representing
    the indices from which you can find the matching elements.

    """

    # compute the L1 distance
    x = np.asarray(x)
    y = np.asarray(y)
    dist = np.abs(np.subtract.outer(x, y))
    if simcarn:
        dist = simcarn_diff(dist)
    dist = dist < th

    # compute the indices
    if broadcast:
        indicesx, indicesy = np.nonzero(dist)
    else:
        matchingx = np.count_nonzero(dist, axis=1)
        indicesx = np.nonzero(matchingx > 0)[0]
        matchingy = np.count_nonzero(dist, axis=0)
        indicesy = np.nonzero(matchingy > 0)[0]

    if return_count:
        # count the elements
        x = indicesx.shape[0]
        y = indicesy.shape[0]
    else:
        # compute the elements
        x = x[indicesx]
        y = y[indicesy]

    if return_indices:
        return x, y, indicesx, indicesy
    else:
        return x, y


def _no_shared_dist(sample1: np.ndarray, sample2: np.ndarray,
                    th: float) -> float:
    intersect = my_intersect(sample1,
                             sample2,
                             th,
                             return_count=True,
                             broadcast=False)
    intersect = min(intersect[0], intersect[1])
    return ((len(sample1) - intersect) + (len(sample2) - intersect)) / 2


def _avg_perc_shared_dist(sample1: np.ndarray, sample2: np.ndarray,
                          th: float) -> float:
    intersect = my_intersect(sample1,
                             sample2,
                             th,
                             return_count=True,
                             broadcast=False)
    intersect = min(intersect[0], intersect[1])
    return 1 - (intersect / len(sample1) + intersect / len(sample2)) / 2


def _min_perc_shared_dist(sample1: np.ndarray, sample2: np.ndarray,
                          th: float) -> float:
    intersect = my_intersect(sample1,
                             sample2,
                             th,
                             return_count=True,
                             broadcast=False)
    intersect = min(intersect[0], intersect[1])
    return 1 - intersect / max(len(sample1), len(sample2))


def _jaccard_dist(sample1: np.ndarray, sample2: np.ndarray, th: float,
                  simcarn: bool) -> float:
    """
    match pitches if their difference lies under `th`
    """

    intersect = my_intersect(sample1,
                             sample2,
                             th,
                             return_count=True,
                             simcarn=simcarn,
                             broadcast=False)
    intersect = min(intersect[0], intersect[1])
    return 1 - intersect / (len(sample1) + len(sample2) - intersect)


def _jaccard_fuzzy_dist(sample1: np.ndarray, sample2: np.ndarray, th: float,
                        simcarn: bool) -> float:
    """
    This function computes pairwise difference between `sample1` and
    `sample2`. Then, counts how many pairs lies under threshold `th`,
    normalize them to `th`, and counts this value instead of 1 for the
    computation of the ratio matched/total_size

    Returns
    -------

    `float` :
        (S - n - d) / S, where:

        * `S` is `sample1.shape[0] * sample2.shape[0]`: the total number of
          pairs
        * `n` is the nmber of paris whos L1 distance is under `th` (matched
          pairs)
        * `d` is the accumulated L1 distance of the matched pairs
    """

    _, _, idx1, idx2 = my_intersect(sample1,
                                    sample2,
                                    th=th,
                                    return_indices=True,
                                    simcarn=simcarn,
                                    broadcast=True)

    n_match = idx1.shape[0]

    dist = np.sum(np.abs(sample1[idx1] - sample2[idx2]) / th)

    size = sample1.shape[0] * sample2.shape[0]
    return (size - n_match + dist) / size
