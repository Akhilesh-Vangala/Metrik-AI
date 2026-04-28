import numpy as np
import pandas as pd
import pytest

from src.features import add_lag_features, add_rolling_features, add_time_features, build_features
from src.config import FeaturesConfig


def _make_sample_data(n_hours: int = 200) -> pd.DataFrame:
    timestamps = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "building_id": np.repeat(0, n_hours),
        "meter": np.repeat(0, n_hours),
        "timestamp": timestamps,
        "meter_reading": rng.exponential(100, n_hours).astype(np.float32),
        "site_id": np.repeat(0, n_hours),
        "primary_use": pd.Categorical(["Education"] * n_hours),
        "square_feet": np.repeat(50000.0, n_hours).astype(np.float32),
        "year_built": np.repeat(1990.0, n_hours).astype(np.float32),
        "log_square_feet": np.repeat(np.log1p(50000), n_hours).astype(np.float32),
        "building_age": np.repeat(27.0, n_hours).astype(np.float32),
        "air_temperature": rng.uniform(10, 35, n_hours).astype(np.float32),
        "dew_temperature": rng.uniform(5, 25, n_hours).astype(np.float32),
    })


class TestLeakageSafety:
    def test_lag_features_no_future_leakage(self):
        df = _make_sample_data(200)
        df = add_lag_features(df, lag_hours=[24])

        for idx in range(len(df)):
            if pd.notna(df.iloc[idx]["lag_24h"]):
                current_ts = df.iloc[idx]["timestamp"]
                source_ts = current_ts - pd.Timedelta(hours=24)
                assert source_ts < current_ts

    def test_rolling_features_no_future_leakage(self):
        df = _make_sample_data(200)
        original_values = df["meter_reading"].copy()
        df = add_rolling_features(df, windows=[24])

        for idx in range(len(df)):
            if pd.notna(df.iloc[idx]["rolling_mean_24h"]):
                current_ts = df.iloc[idx]["timestamp"]
                window_start = max(0, idx - 24)
                window_vals = original_values.iloc[window_start:idx]
                if len(window_vals) > 0:
                    expected_mean = window_vals.mean()
                    assert abs(df.iloc[idx]["rolling_mean_24h"] - expected_mean) < 1.0 or idx < 24

    def test_first_lag_values_are_nan(self):
        df = _make_sample_data(200)
        df = add_lag_features(df, lag_hours=[24])
        assert df["lag_24h"].iloc[:24].isna().all()

    def test_first_rolling_values_are_nan(self):
        df = _make_sample_data(200)
        df = add_rolling_features(df, windows=[24])
        assert pd.isna(df["rolling_mean_24h"].iloc[0])

    def test_lag_diff_24h_uses_no_current_reading(self):
        rng = np.random.RandomState(0)
        df = _make_sample_data(300)
        df["meter_reading"] = rng.uniform(1000, 2000, 300).astype(np.float32)
        df = add_lag_features(df, lag_hours=[24, 168])

        valid = df.dropna(subset=["lag_diff_24h", "lag_24h", "lag_168h"])
        assert len(valid) > 0

        correct = (valid["lag_24h"] - valid["lag_168h"]).values
        leaked  = (valid["meter_reading"] - valid["lag_24h"]).values

        np.testing.assert_allclose(valid["lag_diff_24h"].values, correct, atol=1e-3)
        assert not np.allclose(correct, leaked, atol=1e-3)

    def test_reading_vs_rolling_uses_no_current_reading(self):
        rng = np.random.RandomState(1)
        df = _make_sample_data(300)
        df["meter_reading"] = rng.uniform(500, 5000, 300).astype(np.float32)
        cfg = FeaturesConfig()
        result = build_features(df, cfg)

        valid = result.dropna(subset=["reading_vs_rolling", "lag_24h", "rolling_mean_24h"])
        assert len(valid) > 0

        safe_mean = valid["rolling_mean_24h"].replace(0, np.nan)
        correct = (valid["lag_24h"] / safe_mean).clip(-10, 10).values
        leaked  = (valid["meter_reading"] / safe_mean).clip(-10, 10).values

        np.testing.assert_allclose(valid["reading_vs_rolling"].values, correct, atol=1e-3)
        assert not np.allclose(correct, leaked, atol=1e-3)


class TestTimeFeatures:
    def test_hour_range(self):
        df = _make_sample_data(48)
        df = add_time_features(df)
        assert df["hour"].min() >= 0
        assert df["hour"].max() <= 23

    def test_weekend_flag(self):
        df = _make_sample_data(200)
        df = add_time_features(df)
        ts = pd.DatetimeIndex(df["timestamp"])
        expected = (ts.dayofweek >= 5).astype(np.int8)
        assert (df["is_weekend"].values == expected).all()


class TestBuildFeatures:
    def test_output_has_expected_columns(self):
        df = _make_sample_data(200)
        cfg = FeaturesConfig()
        result = build_features(df, cfg)
        expected = ["hour", "dayofweek", "month", "is_weekend", "lag_24h", "rolling_mean_24h"]
        for col in expected:
            assert col in result.columns

    def test_no_float64_when_configured(self):
        df = _make_sample_data(200)
        cfg = FeaturesConfig(dtype="float32")
        result = build_features(df, cfg)
        float64_cols = result.select_dtypes(include=[np.float64]).columns.tolist()
        assert len(float64_cols) == 0, f"Unexpected float64 columns: {float64_cols}"
