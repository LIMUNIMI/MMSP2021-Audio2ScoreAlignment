# cython: language_level=3
# cython: boundscheck=False
"""
Fast computation of distances with float32 and cython. Distance matrix computed
with thread parallelization!
"""
import numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport sqrt
from libc.math cimport fabs
# from libc.stdlib cimport abs as iabs


cdef float eps = 1e-15

cdef float[:] sub(float[:] A, float c) nogil:
    for i in range(len(A)):
        A[i] -= c

    return A

cdef float norm(float[:] A) nogil:
    cdef float out = 0
    for i in range(len(A)):
        out += A[i]**2

    return sqrt(out) + eps

cdef float dot(float[:] A, float[:] B) nogil:
    cdef float out = 0
    for i in range(len(A)):
        out += A[i] * B[i]

    return out

cdef float mean(float[:] A) nogil:
    cdef float out = 0
    for i in range(len(A)):
        out += A[i]

    return out / len(A)

cdef float ccosine(float[:] A, float[:] B) nogil:
    return 1 - dot(A, B) / (norm(A) * norm(B))

def cosine(A, B):
    return ccosine(A, B)

cdef float chamming(float[:] A, float[:] B) nogil:
    cdef float out = 0
    for i in range(len(A)):
        if A[i] != B[i]:
            out += 1
    return out

def hamming(A, B):
    return chamming(A, B)

cdef float cminkowski(float[:] A, float[:] B, float p) nogil:
    cdef float out = 0
    cdef float normA = 0
    cdef float normB = 0
    for i in range(len(A)):
        out += fabs(A[i] - B[i])**p
    return out**(1.0/p)
    # return out**(1.0/p)

def minkowski(A, B, p):
    return cminkowski(A, B, p)

def euclidean(A, B):
    return cminkowski(A, B, 2)

def manhattan(A, B):
    return cminkowski(A, B, 1)

cdef float cbraycurtis(float[:] A, float[:] B) nogil:
    cdef float num = 0
    cdef float den = 0

    for i in range(len(A)):
        num += fabs(A[i] - B[i])
        den += fabs(A[i] + B[i])

    return num/(den + eps)

def braycurtis(A, B):
    return cbraycurtis(A, B)

cdef float ccanberra(float[:] A, float[:] B) nogil:
    cdef float out = 0

    for i in range(len(A)):
        out += fabs(A[i] - B[i]) / (fabs(A[i]) + fabs(B[i]) + eps)

    return out

def canberra(A, B):
    return ccanberra(A, B)

cdef float cchebyshev(float[:] A, float[:] B) nogil:
    cdef float out = -1

    for i in range(len(A)):
        out = max(out, fabs(A[i] - B[i]))

    return out

def chebyshev(A, B):
    return cchebyshev(A, B)

cdef float ccorrelation(float[:] A, float[:] B) nogil:
    A = sub(A, mean(A))
    B = sub(B, mean(B))
    return ccosine(A, B)

def correlation(A, B):
    return ccorrelation(A, B)

def cdist(float[:, :] A, float[:, :] B, str metric='euclidean', float p=1):
    """
    Compute distance matrix with parallel threading (without GIL).

    Arguments
    ---
    * `A` : array of array -> float32 (np.array)
        First collection of samples. Features are the last dimension.
    * `B` : array of array -> float32 (np.array)
        Second collection of samples. Features are the last dimension.
    * `metric` : str
        A string indicating the metric that should be used.
        Available metrics:
        - 'cosine' : 1 - cosine similarity
        - 'minkowski' : minkowski normalized to 1
        - 'manhattan' : minkowski with p = 1
        - 'euclidean' : minkowski with p = 2
        - 'hamming' : number of different entries
        - 'braycurtis' : bray-curtis distance
        - 'canberra' : canberra distance
        - 'chebyshev' : chebyshev distance
        - 'correlation' : correlation distance
    * `p` : float32
        A value representing the `p` used for minkowski

    Returns
    ---
    * array of array :
        shape: (A.shape[1], B.shape[1])
    """
    cdef int a_cols = A.shape[1]
    cdef int b_cols = B.shape[1]
    cdef float[:, :] out = np.empty((a_cols, b_cols), dtype=np.float32)
    cdef int i = 0
    cdef int j = 0

    for i in prange(a_cols, nogil=True):
        for j in range(b_cols):
            if metric == 'cosine':
                out[i, j] = ccosine(A[:, i], B[:, j])
            elif metric == 'hamming':
                out[i, j] = chamming(A[:, i], B[:, j])
            elif metric == 'manhattan':
                out[i, j] = cminkowski(A[:, i], B[:, j], 1)
            elif metric == 'euclidean':
                out[i, j] = cminkowski(A[:, i], B[:, j], 2)
            elif metric == 'minkowski':
                out[i, j] = cminkowski(A[:, i], B[:, j], p)
            elif metric == 'braycurtis':
                out[i, j] = cbraycurtis(A[:, i], B[:, j])
            elif metric == 'canberra':
                out[i, j] = ccanberra(A[:, i], B[:, j])
            elif metric == 'chebyshev':
                out[i, j] = cchebyshev(A[:, i], B[:, j])
            elif metric == 'correlation':
                out[i, j] = ccorrelation(A[:, i], B[:, j])

    return out
