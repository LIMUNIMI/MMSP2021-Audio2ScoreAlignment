from math import floor
from typing import Callable

import numpy as np
import plotly.express as px
from dtw import stepPattern as sp

# yapf: disable
# DTW
#: a symmetric pattern for DTW
symmetric = sp.StepPattern(
    sp._c(
        # diagonal
        1, 1, 1, -1,
        1, 0, 0, 3,

        # vertical
        2, 1, 0, -1,
        2, 0, 0, 2,

        # horizontal
        3, 0, 1, -1,
        3, 0, 0, 2,

        # 1 vertical + diagonal
        4, 2, 1, -1,
        4, 1, 0, 2,
        4, 0, 0, 2,

        # 1 horizontal + diagonal
        5, 1, 2, -1,
        5, 0, 1, 2,
        5, 0, 0, 2,
    ),
    "NA")

#: an asymmetric pattern which favours the horizontal paths (changing column is
#: easier than changing row)
asymmetric_hor = sp.StepPattern(
    sp._c(
        # diagonal
        1, 1, 1, -1,
        1, 0, 0, 3,

        # vertical
        2, 1, 0, -1,
        2, 0, 0, 2,

        # horizontal
        3, 0, 1, -1,
        3, 0, 0, 1,

        # 1 vertical + diagonal
        4, 2, 1, -1,
        4, 1, 0, 2,
        4, 0, 0, 2,

        # 1 horizontal + diagonal
        5, 1, 2, -1,
        5, 0, 1, 2,
        5, 0, 0, 1,
    ),
    "NA")

#: an asymmetric pattern which favours the vertical paths (changing row is
#: easier than changing column)
asymmetric_ver = sp.StepPattern(
    sp._c(
        # diagonal
        1, 1, 1, -1,
        1, 0, 0, 3,

        # vertical
        2, 1, 0, -1,
        2, 0, 0, 1,

        # horizontal
        3, 0, 1, -1,
        3, 0, 0, 2,

        # 1 vertical + diagonal
        4, 2, 1, -1,
        4, 1, 0, 2,
        4, 0, 0, 1,

        # 1 horizontal + diagonal
        5, 1, 2, -1,
        5, 0, 1, 2,
        5, 0, 0, 2,
    ),
    "NA")

#: an asymmetric pattern which favours the vertical paths (changing row is
#: easier than changing column); this is like dtw.stepPattern.asymmetric, but
#: that one favours horizontal paths
asymmetric1 = sp.StepPattern(
    sp._c(
        # diagonal
        1, 1, 1, -1,
        1, 0, 0, 1,

        # vertical
        2, 0, 1, -1,
        2, 0, 0, 1,

        # second diagonal
        3, 2, 1, -1,
        3, 0, 0, 1
    ),
    "N")
# yapf: enable

symmetric1 = sp.symmetric1
symmetric2 = sp.symmetric2
asymmetric2 = sp.asymmetric

step_patterns = (
    asymmetric_hor,
    asymmetric_ver,
    symmetric,
    symmetric1,
    symmetric2,
    asymmetric1,
    asymmetric2,
)


def avg_dist(dist: Callable, dist_args: dict):
    def new_dist(x: list, y: list):
        out = 0
        for i in range(len(x)):
            out += dist(x[i], y[i], **dist_args)
        return out / len(x)

    return new_dist


def idx_range(idx, radius, length):
    """
    given an idx, a radius and a maximum length, returns starting and ending
    indices of a a window centered at that idx and having that radius, without
    indices > length nor < 0
    """
    return max(0, idx - radius), min(length, idx + radius + 1)


class FedeWindow(object):
    """
    A windowing function which computes a different slanted-band at each point
    based on the local difference of the main slanted diagonal; the local
    radius is computed as:

    `max(
        min_radius,
        floor(
            alpha * avg_dist_fn(
                    x[i - beta : i + beta],
                    y[j - beta : j + beta]
                )
        )
    )`

    where:
    * N is the length of x
    * M is the length of y
    * avg_dist_fn is the average of dist_fn on each corresponding sample
    * j = floor(i * M / N)

    By words, `beta` is half the length of a sliding window used to compute
    distances between excerpts of `x` and `y` taken along the slanted diagonal.
    The distance is multiplied by `alpha` to get the local radius length.

    `x` and `y` are sequences with shape ([M, N], features)
    """
    def __init__(self,
                 x,
                 y,
                 dist_fn: Callable,
                 alpha=5,
                 beta=5,
                 min_radius=5,
                 dist_args: dict = {}):

        self.alpha = alpha
        self.beta = beta
        self.min_radius = min_radius
        self.dist_fn = avg_dist(dist_fn, dist_args)

        self.compute_mask(x, y)

    def compute_mask(self, x, y):
        # take the distance function
        N = len(x)
        M = len(y)
        transpose = False
        if M > N:
            # x should always be longer than y
            x, y = y, x
            N, M = M, N
            # if we swap x and y, we need to swap the mask too
            transpose = True

        # a mask to remember points
        self.mask = np.zeros((len(x), len(y)), dtype=np.bool8)

        # for each point in x
        for i in range(N):
            # compute the point in y along the diagonal
            j = floor(i * M / N)

            # compute the sliding windows
            start_x, end_x = idx_range(i, self.beta, N)
            start_y, end_y = idx_range(j, self.beta, M)
            _x = x[start_x:end_x]
            _y = y[start_y:end_y]

            # pad the windows
            if start_x == 0:
                _x = [[0]] * (self.beta - i) + _x
            elif end_x == N:
                _x = _x + [[0]] * (i + self.beta - N)
            if start_y == 0:
                _y = [[0]] * (self.beta - j) + _y
            elif end_y == M:
                _y = _y + [[0]] * (j + self.beta - M)

            # compute the local radius
            lr = max(self.min_radius,
                     floor(self.alpha * self.dist_fn(_x, _y)))

            # set the points inside the local radius to True
            self.mask[slice(*idx_range(i, lr, N)),
                      slice(*idx_range(j, lr, M))] = True

        if transpose:
            self.mask = self.mask.T

    def __call__(self, i, j, query_size=None, reference_size=None):
        return self.mask[i, j]

    def plot(self):
        """
        Return a plotly Figure object representing the heatmap of the mask
        """
        return px.imshow(self.mask, aspect='auto')


def _remove_conflicting_match(arr_x: np.ndarray, arr_y: np.ndarray,
                              graph_matrix: np.ndarray, target: int):
    """
    1. look for repeated values in `arr_x` or `arr_y`, depending on `target`
    2. look for the maximum value in `graph_matrix[1]`, at the indices in
    `arr_x` and `arr_y` relative to the repeated values
    3. among the repeated values in the target, chose the ones corresponding to
    the maximum in `graps_matrix[1]`
    4. return `arr_x` and `arr_y` without the removed indices
    """
    if target == 0:
        _target = arr_x
    elif target == 1:
        _target = arr_y
    else:
        raise RuntimeError(f"`target` should be 0 or 1, used {target} instead")

    arr_mask = np.ones(_target.shape[0], dtype=np.bool8)
    unique_vals, unique_count = np.unique(_target, return_counts=True)
    for unique_val in unique_vals[unique_count > 1]:
        conflicting_idx = np.nonzero(_target == unique_val)[0]
        to_keep_idx_of_idx = np.argmax(graph_matrix[1, arr_x[conflicting_idx],
                                                    arr_y[conflicting_idx]])
        arr_mask[conflicting_idx] = 0
        arr_mask[conflicting_idx[to_keep_idx_of_idx]] = 1

    return arr_x[arr_mask], arr_y[arr_mask]


def merge_matching_indices(args):
    """
    Takes a list of mapping indices, fills the graph matrix counting the number
    of times a match happens in the mappings. Then start taking matching from
    the most matched and iteratively adding new matching. If two conflicting
    matching have the same number of counts, takes the matching which appears
    in the longest mapping; in case of parity the first one is taken
    """

    # creating the matrix
    num_notes = np.max([arg[:, 0].max() for arg in args]) + 1, np.max(
        [arg[:, 1].max() for arg in args]) + 1
    # dim 0 records the counts, dim 1 records the most long mapping containing
    # the matching
    graph_matrix = np.zeros((2, num_notes[0], num_notes[1]), dtype=np.int64)

    # filling the matrix
    for arg in args:
        # the count
        graph_matrix[0, arg[:, 0], arg[:, 1]] += 1

        # the length
        L = arg.shape[0]
        graph_matrix[1, arg[:, 0], arg[:, 1]] = np.maximum(
            graph_matrix[1, arg[:, 0], arg[:, 1]], L)

    # merging
    # two indices which records references to the original matrix
    index_rows = np.arange(num_notes[0])
    index_cols = np.arange(num_notes[1])
    merged = []
    for k in range(len(args), 0, -1):
        # take matchings that appear `k` times
        candidates_row, candidates_col = np.nonzero(graph_matrix[0] == k)

        # remove conflicting candidates
        candidates_row, candidates_col = _remove_conflicting_match(
            candidates_row, candidates_col, graph_matrix, 0)
        candidates_row, candidates_col = _remove_conflicting_match(
            candidates_row, candidates_col, graph_matrix, 1)

        # add candidates to the output
        merged.append(
            np.stack([index_rows[candidates_row], index_cols[candidates_col]],
                     axis=1))

        # remove matched notes from graph_matrix
        mask_rows = np.ones(graph_matrix.shape[1], dtype=np.bool8)
        mask_cols = np.ones(graph_matrix.shape[2], dtype=np.bool8)
        mask_rows[candidates_row] = 0
        mask_cols[candidates_col] = 0
        graph_matrix = graph_matrix[:, mask_rows]
        graph_matrix = graph_matrix[:, :, mask_cols]

        # remove matched notes from the index
        index_rows = index_rows[mask_rows]
        index_cols = index_cols[mask_cols]

    # re-sort everything and return
    merged = np.concatenate(merged, axis=0)
    # print(f"Added notes from merging: {len(ref) - L}")
    return merged[merged[:, 0].argsort()]
