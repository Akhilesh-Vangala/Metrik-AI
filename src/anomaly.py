from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.config import AnomalyConfig

logger = logging.getLogger(__name__)


def modified_zscore_naive(residuals: np.ndarray) -> np.ndarray:
    """Pure Python loop — intentionally slow for benchmarking."""
    n = len(residuals)
    scores = np.empty(n, dtype=np.float64)

    sorted_r = sorted(residuals)
    mid = n // 2
    if n % 2 == 0:
        median_val = (sorted_r[mid - 1] + sorted_r[mid]) / 2.0
    else:
        median_val = sorted_r[mid]

    abs_devs = [abs(residuals[i] - median_val) for i in range(n)]
    sorted_devs = sorted(abs_devs)
    if n % 2 == 0:
        mad_val = (sorted_devs[mid - 1] + sorted_devs[mid]) / 2.0
    else:
        mad_val = sorted_devs[mid]

    if mad_val == 0:
        mad_val = 1e-10

    for i in range(n):
        scores[i] = 0.6745 * (residuals[i] - median_val) / mad_val

    return scores


def modified_zscore_vectorized(residuals: np.ndarray) -> np.ndarray:
    """Fully vectorized NumPy implementation."""
    median_val = np.median(residuals)
    mad = np.median(np.abs(residuals - median_val))
    if mad == 0:
        mad = 1e-10
    return (0.6745 * (residuals - median_val) / mad).astype(np.float32)


def detect_anomalies(
    df: pd.DataFrame,
    residual_col: str = "residual",
    cfg: AnomalyConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = AnomalyConfig()

    t0 = time.perf_counter()
    logger.info("Running per-meter anomaly detection (method=%s, threshold=%.1f)", cfg.method, cfg.threshold)

    df = df.copy()

    def _score_group(group: pd.DataFrame) -> pd.Series:
        residuals = group[residual_col].values.astype(np.float64)
        if len(residuals) < 10:
            return pd.Series(np.zeros(len(residuals)), index=group.index, dtype=np.float32)
        return pd.Series(modified_zscore_vectorized(residuals), index=group.index)

    df["anomaly_score"] = df.groupby(["building_id", "meter"], group_keys=False).apply(_score_group, include_groups=False)
    df["is_anomaly"] = (np.abs(df["anomaly_score"]) > cfg.threshold).astype(np.int8)

    df = _add_temporal_clusters(df)

    elapsed = time.perf_counter() - t0
    n_flagged = df["is_anomaly"].sum()
    pct = 100 * n_flagged / len(df) if len(df) > 0 else 0
    logger.info("Anomalies: %d / %d (%.2f%%) in %.3fs", n_flagged, len(df), pct, elapsed)

    return df


def detect_anomalies_global(
    df: pd.DataFrame,
    residual_col: str = "residual",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Global scoring (for benchmarking speed — not for production use)."""
    df = df.copy()
    residuals = df[residual_col].values.astype(np.float32)
    df["anomaly_score"] = modified_zscore_vectorized(residuals)
    df["is_anomaly"] = (np.abs(df["anomaly_score"]) > threshold).astype(np.int8)
    return df


def _add_temporal_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Tag consecutive anomalous hours — sustained anomalies are more concerning."""
    df = df.sort_values(["building_id", "meter", "timestamp"])

    def _cluster_streak(s: pd.Series) -> pd.Series:
        shifted = s != s.shift()
        group_id = shifted.cumsum()
        return s.groupby(group_id).transform("sum") * s

    df["anomaly_streak"] = df.groupby(["building_id", "meter"])["is_anomaly"].transform(_cluster_streak).astype(np.int16)
    return df


def aggregate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["building_id", "meter"]).agg(
        total_hours=("is_anomaly", "count"),
        anomaly_hours=("is_anomaly", "sum"),
        mean_anomaly_score=("anomaly_score", "mean"),
        max_anomaly_score=("anomaly_score", lambda s: s.abs().max()),
        mean_residual=("residual", "mean"),
        total_excess=("residual", lambda s: s.clip(lower=0).sum()),
        max_streak=("anomaly_streak", "max"),
    ).reset_index()

    agg["anomaly_rate"] = (agg["anomaly_hours"] / agg["total_hours"]).astype(np.float32)
    return agg.sort_values("anomaly_rate", ascending=False)
