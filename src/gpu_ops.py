from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def modified_zscore_gpu(residuals: np.ndarray) -> np.ndarray:
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy not available; install with: pip install cupy-cuda12x")

    t0 = time.perf_counter()

    d_residuals = cp.asarray(residuals.astype(np.float64))
    median_val = cp.median(d_residuals)
    mad = cp.median(cp.abs(d_residuals - median_val))

    if float(mad) == 0.0:
        mad = cp.float64(1e-10)

    scores = 0.6745 * (d_residuals - median_val) / mad
    result = cp.asnumpy(scores).astype(np.float32)

    elapsed = time.perf_counter() - t0
    logger.info("GPU anomaly scoring: %d values in %.4fs", len(residuals), elapsed)
    return result


def elementwise_residuals_gpu(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy not available")

    d_actual = cp.asarray(actual.astype(np.float64))
    d_predicted = cp.asarray(predicted.astype(np.float64))
    d_residuals = d_actual - d_predicted
    return cp.asnumpy(d_residuals).astype(np.float32)


def rolling_mean_gpu(values: np.ndarray, window: int) -> np.ndarray:
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy not available")

    d_values = cp.asarray(values.astype(np.float64))
    n = len(d_values)

    valid_mask = ~cp.isnan(d_values)
    safe_values = cp.where(valid_mask, d_values, 0.0)

    cum_sum = cp.cumsum(safe_values)
    cum_count = cp.cumsum(valid_mask.astype(cp.float64))

    idx = cp.arange(n)
    shifted_sum = cp.where(idx >= window, cum_sum[idx - window], 0.0)
    shifted_count = cp.where(idx >= window, cum_count[idx - window], 0.0)

    window_sum = cum_sum - shifted_sum
    window_count = cum_count - shifted_count

    result = cp.where(window_count > 0, window_sum / window_count, cp.nan)

    return cp.asnumpy(result).astype(np.float32)


def check_gpu_status() -> dict:
    info = {"gpu_available": GPU_AVAILABLE}

    if GPU_AVAILABLE:
        try:
            device = cp.cuda.Device(0)
            info["device_name"] = device.attributes.get("DeviceName", "unknown")
            mem = device.mem_info
            info["total_memory_gb"] = round(mem[1] / 1e9, 2)
            info["free_memory_gb"] = round(mem[0] / 1e9, 2)
        except Exception as e:
            info["error"] = str(e)

    return info
