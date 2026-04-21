import numpy as np
import pandas as pd
import pytest

from src.anomaly import modified_zscore_vectorized, modified_zscore_naive, detect_anomalies, aggregate_anomalies
from src.model import time_based_split, train_baseline_mean, compute_residuals
from src.decision import build_audit_list
from src.config import AnomalyConfig


def _make_pipeline_data(n_buildings: int = 5, n_hours: int = 3000) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    for bid in range(n_buildings):
        timestamps = pd.date_range("2016-01-01", periods=n_hours, freq="h")
        base_load = rng.uniform(50, 500)
        readings = base_load + rng.normal(0, base_load * 0.1, n_hours)
        for i in range(n_hours):
            rows.append({
                "building_id": bid,
                "meter": 0,
                "timestamp": timestamps[i],
                "meter_reading": max(0, readings[i]),
                "site_id": bid % 3,
                "primary_use": "Education",
                "square_feet": 50000.0 + bid * 10000,
                "lag_24h": readings[i - 24] if i >= 24 else np.nan,
            })
    return pd.DataFrame(rows)


class TestAnomalyDetection:
    def test_vectorized_matches_naive(self):
        rng = np.random.RandomState(99)
        data = rng.randn(10_000).astype(np.float64)
        naive = modified_zscore_naive(data)
        vectorized = modified_zscore_vectorized(data)
        np.testing.assert_allclose(naive, vectorized, atol=1e-4)

    def test_anomaly_flags_threshold(self):
        data = np.array([0.0, 0.1, -0.1, 10.0, -10.0, 0.05], dtype=np.float64)
        scores = modified_zscore_vectorized(data)
        flags = (np.abs(scores) > 3.5).astype(int)
        assert flags[3] == 1
        assert flags[4] == 1
        assert flags[0] == 0


class TestTimeBasedSplit:
    def test_no_temporal_overlap(self):
        df = _make_pipeline_data(2, 3000)
        train, val = time_based_split(df, validation_months=1)
        assert train["timestamp"].max() < val["timestamp"].min()

    def test_all_rows_accounted(self):
        df = _make_pipeline_data(2, 3000)
        train, val = time_based_split(df, validation_months=1)
        assert len(train) + len(val) == len(df)


class TestEndToEnd:
    def test_mini_pipeline(self):
        df = _make_pipeline_data(3, 3000)
        train, val = time_based_split(df, validation_months=1)

        baseline = train_baseline_mean(train, val)
        assert baseline.rmse > 0
        assert baseline.rmse < 1e6

        residuals = compute_residuals(val["meter_reading"].values, baseline.predictions)
        val = val.copy()
        val["residual"] = residuals

        cfg = AnomalyConfig(threshold=3.0)
        val = detect_anomalies(val, cfg=cfg)
        assert "is_anomaly" in val.columns
        assert "anomaly_score" in val.columns

        summary = aggregate_anomalies(val)
        assert len(summary) > 0

        meta = pd.DataFrame({
            "building_id": range(3),
            "site_id": [0, 1, 2],
            "primary_use": ["Education"] * 3,
            "square_feet": [50000, 60000, 70000],
        })

        audit = build_audit_list(summary, meta, min_hours=10)
        assert "rank" in audit.columns
        assert "priority_score" in audit.columns
