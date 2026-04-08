from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import holidays
import numpy as np
import pandas as pd

from src.config import FeaturesConfig

logger = logging.getLogger(__name__)

US_HOLIDAYS = holidays.US()


@lru_cache(maxsize=1024)
def _is_holiday(date: str) -> bool:
    return date in US_HOLIDAYS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.DatetimeIndex(df["timestamp"])
    df["hour"] = ts.hour.astype(np.int8)
    df["dayofweek"] = ts.dayofweek.astype(np.int8)
    df["month"] = ts.month.astype(np.int8)
    df["is_weekend"] = (ts.dayofweek >= 5).astype(np.int8)
    df["day_of_year"] = ts.dayofyear.astype(np.int16)

    date_strs = ts.strftime("%Y-%m-%d")
    df["is_holiday"] = np.array(
        [_is_holiday(d) for d in date_strs], dtype=np.int8
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)

    return df


def add_lag_features(
    df: pd.DataFrame,
    lag_hours: Sequence[int] = (24, 168),
    target_col: str = "meter_reading",
) -> pd.DataFrame:
    df = df.sort_values(["building_id", "meter", "timestamp"]).copy()
    group_key = ["building_id", "meter"]

    for lag in lag_hours:
        col_name = f"lag_{lag}h"
        df[col_name] = (
            df.groupby(group_key)[target_col]
            .shift(lag)
            .astype(np.float32)
        )

    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (24, 168),
    target_col: str = "meter_reading",
) -> pd.DataFrame:
    df = df.sort_values(["building_id", "meter", "timestamp"]).copy()
    group_key = ["building_id", "meter"]

    for w in windows:
        col_mean = f"rolling_mean_{w}h"
        col_std = f"rolling_std_{w}h"

        grouped = df.groupby(group_key)[target_col]
        rolling = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 4)).mean()
        )
        df[col_mean] = rolling.astype(np.float32)

        rolling_s = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 4)).std()
        )
        df[col_std] = rolling_s.astype(np.float32)

    return df


def add_building_features(df: pd.DataFrame) -> pd.DataFrame:
    if "primary_use" in df.columns and df["primary_use"].dtype.name == "category":
        df["primary_use_code"] = df["primary_use"].cat.codes.astype(np.int8)
    elif "primary_use" in df.columns:
        df["primary_use_code"] = pd.Categorical(df["primary_use"]).codes.astype(np.int8)

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    if "air_temperature" in df.columns:
        df["temp_diff"] = (
            df["air_temperature"] - df["dew_temperature"]
        ).astype(np.float32)

    return df


def build_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    logger.info("Building features for %d rows", len(df))

    df = add_time_features(df)
    df = add_lag_features(df, lag_hours=cfg.lag_hours)
    df = add_rolling_features(df, windows=cfg.rolling_windows)
    df = add_building_features(df)
    df = add_weather_features(df)

    if cfg.dtype == "float32":
        float_cols = df.select_dtypes(include=[np.float64]).columns
        df[float_cols] = df[float_cols].astype(np.float32)

    return df


def build_features_naive(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Row-by-row feature engineering for benchmarking against the vectorized version."""
    df = df.sort_values(["building_id", "meter", "timestamp"]).copy()

    hours = []
    dows = []
    months = []
    weekends = []

    for _, row in df.iterrows():
        ts = row["timestamp"]
        hours.append(ts.hour)
        dows.append(ts.dayofweek)
        months.append(ts.month)
        weekends.append(1 if ts.dayofweek >= 5 else 0)

    df["hour"] = hours
    df["dayofweek"] = dows
    df["month"] = months
    df["is_weekend"] = weekends

    return df


FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "day_of_year", "hour_sin", "hour_cos",
    "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h", "rolling_std_168h",
    "log_square_feet", "building_age",
    "primary_use_code",
    "air_temperature", "dew_temperature", "temp_diff",
    "site_id", "meter",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]
