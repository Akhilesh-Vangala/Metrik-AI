# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt


def rolling_mean_cython(
    np.ndarray[np.float64_t, ndim=1] values,
    int window
):
    cdef int n = values.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double total, v
    cdef int i, j, start, count, valid

    for i in range(n):
        if i < window:
            start = 0
            count = i
        else:
            start = i - window
            count = window

        if count == 0:
            result[i] = np.nan
            continue

        total = 0.0
        valid = 0
        for j in range(start, start + count):
            v = values[j]
            if v == v:  # NaN check
                total += v
                valid += 1

        if valid > 0:
            result[i] = total / valid
        else:
            result[i] = np.nan

    return result


def rolling_std_cython(
    np.ndarray[np.float64_t, ndim=1] values,
    int window
):
    cdef int n = values.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double total, mean_val, sq_sum, v
    cdef int i, j, start, count, valid

    for i in range(n):
        if i < window:
            start = 0
            count = i
        else:
            start = i - window
            count = window

        if count < 2:
            result[i] = np.nan
            continue

        total = 0.0
        valid = 0
        for j in range(start, start + count):
            v = values[j]
            if v == v:
                total += v
                valid += 1

        if valid < 2:
            result[i] = np.nan
            continue

        mean_val = total / valid
        sq_sum = 0.0
        for j in range(start, start + count):
            v = values[j]
            if v == v:
                sq_sum += (v - mean_val) * (v - mean_val)

        result[i] = sqrt(sq_sum / (valid - 1))

    return result


def modified_zscore_cython(np.ndarray[np.float64_t, ndim=1] residuals):
    cdef int n = residuals.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] scores = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] sorted_r
    cdef np.ndarray[np.float64_t, ndim=1] abs_devs
    cdef double median_val, mad_val
    cdef int mid, i

    sorted_r = np.sort(residuals)
    mid = n // 2
    if n % 2 == 0:
        median_val = (sorted_r[mid - 1] + sorted_r[mid]) / 2.0
    else:
        median_val = sorted_r[mid]

    abs_devs = np.empty(n, dtype=np.float64)
    for i in range(n):
        abs_devs[i] = fabs(residuals[i] - median_val)

    abs_devs = np.sort(abs_devs)
    if n % 2 == 0:
        mad_val = (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
    else:
        mad_val = abs_devs[mid]

    if mad_val == 0.0:
        mad_val = 1e-10

    for i in range(n):
        scores[i] = 0.6745 * (residuals[i] - median_val) / mad_val

    return scores
