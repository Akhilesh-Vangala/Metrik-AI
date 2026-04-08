from __future__ import annotations

import numpy as np
import numba
from numba import njit, prange


@njit(cache=True)
def _median_sorted(arr: np.ndarray) -> float:
    n = len(arr)
    mid = n // 2
    if n % 2 == 0:
        return (arr[mid - 1] + arr[mid]) / 2.0
    return arr[mid]


@njit(parallel=True, cache=True)
def modified_zscore_numba(residuals: np.ndarray) -> np.ndarray:
    n = len(residuals)
    sorted_r = np.sort(residuals)
    median_val = _median_sorted(sorted_r)

    abs_devs = np.empty(n, dtype=np.float64)
    for i in prange(n):
        abs_devs[i] = abs(residuals[i] - median_val)

    sorted_devs = np.sort(abs_devs)
    mad_val = _median_sorted(sorted_devs)

    if mad_val == 0.0:
        mad_val = 1e-10

    scores = np.empty(n, dtype=np.float64)
    for i in prange(n):
        scores[i] = 0.6745 * (residuals[i] - median_val) / mad_val

    return scores


@njit(parallel=True, cache=True)
def rolling_mean_numba(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
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
            if not np.isnan(v):
                total += v
                valid += 1

        result[i] = total / valid if valid > 0 else np.nan

    return result


@njit(parallel=True, cache=True)
def rolling_std_numba(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
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
            if not np.isnan(v):
                total += v
                valid += 1

        if valid < 2:
            result[i] = np.nan
            continue

        mean_val = total / valid
        sq_sum = 0.0
        for j in range(start, start + count):
            v = values[j]
            if not np.isnan(v):
                sq_sum += (v - mean_val) ** 2

        result[i] = np.sqrt(sq_sum / (valid - 1))

    return result


@njit(parallel=True, cache=True)
def flag_anomalies_numba(
    scores: np.ndarray, threshold: float
) -> np.ndarray:
    n = len(scores)
    flags = np.empty(n, dtype=np.int8)
    for i in prange(n):
        flags[i] = 1 if abs(scores[i]) > threshold else 0
    return flags


def warmup():
    dummy = np.random.randn(1000).astype(np.float64)
    modified_zscore_numba(dummy)
    rolling_mean_numba(dummy, 24)
    rolling_std_numba(dummy, 24)
    flag_anomalies_numba(dummy, 3.5)
