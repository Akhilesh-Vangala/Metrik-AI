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
    df["is_holiday"] = np.array([_is_holiday(d) for d in date_strs], dtype=np.int8)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

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

    df["lag_diff_24h"] = (df[target_col] - df.get("lag_24h", 0)).astype(np.float32)

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
        col_min = f"rolling_min_{w}h"
        col_max = f"rolling_max_{w}h"

        grouped = df.groupby(group_key)[target_col]

        df[col_mean] = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 4)).mean()
        ).astype(np.float32)

        df[col_std] = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 4)).std()
        ).astype(np.float32)

        df[col_min] = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 4)).min()
        ).astype(np.float32)

        df[col_max] = grouped.transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 4)).max()
        ).astype(np.float32)

    return df


def add_building_features(df: pd.DataFrame) -> pd.DataFrame:
    if "primary_use" in df.columns and df["primary_use"].dtype.name == "category":
        df["primary_use_code"] = df["primary_use"].cat.codes.astype(np.int8)
    elif "primary_use" in df.columns:
        df["primary_use_code"] = pd.Categorical(df["primary_use"]).codes.astype(np.int8)

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    if "air_temperature" not in df.columns:
        return df

    df["temp_diff"] = (df["air_temperature"] - df["dew_temperature"]).astype(np.float32)
    df["relative_humidity"] = np.clip(
        100 * np.exp(17.625 * df["dew_temperature"] / (243.04 + df["dew_temperature"]))
        / np.exp(17.625 * df["air_temperature"] / (243.04 + df["air_temperature"])),
        0, 100,
    ).astype(np.float32)

    df["temp_squared"] = (df["air_temperature"] ** 2).astype(np.float32)
    if "wind_speed" in df.columns:
        ws = df["wind_speed"].clip(lower=0.1)
        df["wind_chill"] = np.where(
            df["air_temperature"] < 10,
            13.12 + 0.6215 * df["air_temperature"]
            - 11.37 * (ws ** 0.16)
            + 0.3965 * df["air_temperature"] * (ws ** 0.16),
            df["air_temperature"],
        ).astype(np.float32)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" in df.columns and "is_weekend" in df.columns:
        df["hour_weekend"] = (df["hour"] * df["is_weekend"]).astype(np.int16)

    if "air_temperature" in df.columns and "hour" in df.columns:
        df["temp_x_hour"] = (df["air_temperature"] * df["hour"]).astype(np.float32)

    if "log_square_feet" in df.columns and "air_temperature" in df.columns:
        df["sqft_x_temp"] = (df["log_square_feet"] * df["air_temperature"]).astype(np.float32)

    if "rolling_mean_24h" in df.columns:
        mean_24 = df["rolling_mean_24h"]
        current = df.get("meter_reading", mean_24)
        safe_mean = mean_24.replace(0, np.nan)
        df["reading_vs_rolling"] = (current / safe_mean).clip(-10, 10).astype(np.float32)

    return df


def build_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    logger.info("Building features for %d rows", len(df))

    df = add_time_features(df)
    df = add_lag_features(df, lag_hours=cfg.lag_hours)
    df = add_rolling_features(df, windows=cfg.rolling_windows)
    df = add_building_features(df)
    df = add_weather_features(df)
    df = add_interaction_features(df)

    if cfg.dtype == "float32":
        float_cols = df.select_dtypes(include=[np.float64]).columns
        df[float_cols] = df[float_cols].astype(np.float32)

    return df


def build_features_naive(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Full row-by-row feature engineering — intentionally slow for benchmarking."""
    df = df.sort_values(["building_id", "meter", "timestamp"]).copy()

    hours, dows, months, weekends, holidays_list = [], [], [], [], []
    hour_sins, hour_coss = [], []

    for _, row in df.iterrows():
        ts = row["timestamp"]
        h = ts.hour
        hours.append(h)
        dows.append(ts.dayofweek)
        months.append(ts.month)
        weekends.append(1 if ts.dayofweek >= 5 else 0)
        holidays_list.append(1 if _is_holiday(ts.strftime("%Y-%m-%d")) else 0)
        hour_sins.append(float(np.sin(2 * np.pi * h / 24)))
        hour_coss.append(float(np.cos(2 * np.pi * h / 24)))

    df["hour"] = hours
    df["dayofweek"] = dows
    df["month"] = months
    df["is_weekend"] = weekends
    df["is_holiday"] = holidays_list
    df["hour_sin"] = hour_sins
    df["hour_cos"] = hour_coss

    lag_vals = {lag: [] for lag in cfg.lag_hours}
    groups = df.groupby(["building_id", "meter"])
    for (bid, meter), group in groups:
        readings = group["meter_reading"].tolist()
        for lag in cfg.lag_hours:
            for i in range(len(readings)):
                lag_vals[lag].append(readings[i - lag] if i >= lag else np.nan)

    for lag in cfg.lag_hours:
        df[f"lag_{lag}h"] = lag_vals[lag]

    return df


FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "day_of_year", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "lag_24h", "lag_168h", "lag_diff_24h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_min_24h", "rolling_max_24h",
    "rolling_mean_168h", "rolling_std_168h", "rolling_min_168h", "rolling_max_168h",
    "log_square_feet", "building_age",
    "primary_use_code",
    "air_temperature", "dew_temperature", "temp_diff",
    "relative_humidity", "temp_squared", "wind_chill",
    "cloud_coverage", "wind_speed",
    "hour_weekend", "temp_x_hour", "sqft_x_temp", "reading_vs_rolling",
    "site_id", "meter",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in df.columns]
