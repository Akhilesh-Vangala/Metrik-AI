from __future__ import annotations

import itertools
import logging
import operator
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import AnomalyConfig

logger = logging.getLogger(__name__)


def modified_zscore_naive(residuals: np.ndarray) -> np.ndarray:
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
        import math
        std_val = math.sqrt(sum((r - median_val) ** 2 for r in residuals) / n)
        mad_val = std_val * 1.4826
    if mad_val == 0:
        return np.zeros(n, dtype=np.float64)

    for i in range(n):
        scores[i] = 0.6745 * (residuals[i] - median_val) / mad_val

    return scores


def modified_zscore_vectorized(residuals: np.ndarray) -> np.ndarray:
    median_val = np.median(residuals)
    mad = np.median(np.abs(residuals - median_val))
    if mad == 0:
        mad = np.std(residuals) * 1.4826
    if mad == 0:
        return np.zeros(len(residuals), dtype=np.float32)
    return (0.6745 * (residuals - median_val) / mad).astype(np.float32)


def _get_scorer():
    try:
        from src.numba_ops import modified_zscore_numba, warmup
        warmup()
        logger.info("Using Numba JIT for anomaly scoring")
        return modified_zscore_numba
    except Exception:
        logger.info("Numba unavailable, falling back to NumPy vectorized scoring")
        return modified_zscore_vectorized


_scorer = None


def _score_array(residuals: np.ndarray) -> np.ndarray:
    global _scorer
    if _scorer is None:
        _scorer = _get_scorer()
    return _scorer(residuals.astype(np.float64)).astype(np.float32)


def detect_anomalies(
    df: pd.DataFrame,
    residual_col: str = "residual",
    cfg: AnomalyConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = AnomalyConfig()

    t0 = time.perf_counter()
    logger.info("Per-meter anomaly detection (threshold=%.1f)", cfg.threshold)

    df = df.copy()

    def _score_group(group: pd.DataFrame) -> pd.Series:
        residuals = group[residual_col].values.astype(np.float64)
        if len(residuals) < 10:
            return pd.Series(np.zeros(len(residuals)), index=group.index, dtype=np.float32)
        return pd.Series(_score_array(residuals), index=group.index)

    df["anomaly_score"] = df.groupby(["building_id", "meter"], group_keys=False).apply(
        _score_group, include_groups=False
    )
    df["is_anomaly"] = (np.abs(df["anomaly_score"]) > cfg.threshold).astype(np.int8)
    df = _add_temporal_clusters(df)

    elapsed = time.perf_counter() - t0
    n_flagged = df["is_anomaly"].sum()
    pct = 100 * n_flagged / len(df) if len(df) > 0 else 0
    logger.info("Anomalies: %d / %d (%.2f%%) in %.3fs", n_flagged, len(df), pct, elapsed)

    return df



def _add_temporal_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["building_id", "meter", "timestamp"])

    def _cluster_streak(s: pd.Series) -> pd.Series:
        shifted = s != s.shift()
        group_id = shifted.cumsum()
        return s.groupby(group_id).transform("sum") * s

    df["anomaly_streak"] = df.groupby(
        ["building_id", "meter"]
    )["is_anomaly"].transform(_cluster_streak).astype(np.int16)
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


def summarize_anomalies_by_site(df: pd.DataFrame) -> dict[int, dict]:
    site_counts: defaultdict[int, list[float]] = defaultdict(list)

    for row in df.loc[df["is_anomaly"] == 1].itertuples(index=False):
        site_counts[operator.attrgetter("site_id")(row)].append(
            operator.attrgetter("anomaly_score")(row)
        )

    summary = {}
    for site_id, scores in sorted(site_counts.items(), key=operator.itemgetter(0)):
        abs_scores = list(map(abs, scores))
        cumulative = list(itertools.accumulate(abs_scores, operator.add))
        summary[site_id] = {
            "n_anomalies": len(scores),
            "mean_severity": float(np.mean(abs_scores)),
            "cumulative_severity": cumulative[-1] if cumulative else 0.0,
        }

    return summary
