from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

from src.config import AppConfig

logger = logging.getLogger(__name__)

METER_TYPES = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

TRAIN_DTYPES = {
    "building_id": np.int16,
    "meter": np.int8,
    "meter_reading": np.float32,
}

META_DTYPES = {
    "site_id": np.int8,
    "building_id": np.int16,
    "square_feet": np.float32,
    "year_built": np.float32,
    "floor_count": np.float32,
}

WEATHER_DTYPES = {
    "site_id": np.int8,
    "air_temperature": np.float32,
    "dew_temperature": np.float32,
    "cloud_coverage": np.float32,
    "precip_depth_1_hr": np.float32,
    "sea_level_pressure": np.float32,
    "wind_direction": np.float32,
    "wind_speed": np.float32,
}

SITE0_KBTU_TO_KWH = 0.293071


def load_building_metadata(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.paths.meta_path()
    logger.info("Loading building metadata from %s", path)
    meta = pd.read_csv(path, dtype=META_DTYPES)
    meta["primary_use"] = meta["primary_use"].astype("category")
    meta["building_age"] = (2017 - meta["year_built"]).astype(np.float32)
    meta["log_square_feet"] = np.log1p(meta["square_feet"]).astype(np.float32)
    return meta


def load_weather(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.paths.weather_path()
    logger.info("Loading weather data from %s", path)
    weather = pd.read_csv(path, dtype=WEATHER_DTYPES, parse_dates=["timestamp"])

    for col in ["air_temperature", "dew_temperature", "wind_speed", "sea_level_pressure"]:
        if col in weather.columns:
            weather[col] = weather.groupby("site_id")[col].transform(
                lambda s: s.ffill().bfill()
            )

    weather["cloud_coverage"] = weather["cloud_coverage"].fillna(
        weather["cloud_coverage"].median()
    )
    weather["precip_depth_1_hr"] = weather["precip_depth_1_hr"].fillna(0.0)
    weather["wind_direction"] = weather["wind_direction"].fillna(0.0)

    return weather


def _build_weather_lookup(weather: pd.DataFrame) -> dict[tuple[int, pd.Timestamp], dict]:
    lookup: dict = {}
    cols = [c for c in weather.columns if c not in ("site_id", "timestamp")]
    for row in weather.itertuples(index=False):
        lookup[(row.site_id, row.timestamp)] = {c: getattr(row, c) for c in cols}
    return lookup


def _clean_readings(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    neg_mask = df["meter_reading"] < 0
    n_neg = neg_mask.sum()
    if n_neg > 0:
        logger.warning("Dropping %d negative readings (%.2f%%)", n_neg, 100 * n_neg / n_before)
        df = df[~neg_mask]

    if "site_id" in df.columns:
        site0_elec = (df["site_id"] == 0) & (df["meter"] == 0)
        n_converted = site0_elec.sum()
        if n_converted > 0:
            df.loc[site0_elec, "meter_reading"] = (
                df.loc[site0_elec, "meter_reading"] * SITE0_KBTU_TO_KWH
            ).astype(np.float32)
            logger.info("Converted %d Site 0 electricity readings from kBTU to kWh", n_converted)

    return df


def remove_outliers(df: pd.DataFrame, target_col: str = "meter_reading", cap_quantile: float = 0.999) -> pd.DataFrame:
    n_before = len(df)
    caps = df.groupby(["building_id", "meter"])[target_col].transform("quantile", cap_quantile)
    outlier_mask = df[target_col] > caps
    n_outliers = outlier_mask.sum()

    if n_outliers > 0:
        df.loc[outlier_mask, target_col] = caps[outlier_mask].astype(np.float32)
        logger.info("Capped %d outlier readings (%.2f%%) at %.1f quantile", n_outliers, 100 * n_outliers / n_before, cap_quantile)

    return df


def detect_zero_streaks(df: pd.DataFrame, target_col: str = "meter_reading", min_streak: int = 48) -> pd.DataFrame:
    df = df.sort_values(["building_id", "meter", "timestamp"])
    is_zero = (df[target_col] == 0).astype(int)

    group_key = ["building_id", "meter"]
    streaks = is_zero.groupby([df["building_id"], df["meter"]]).transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).transform("sum")
    )

    bad_mask = (is_zero == 1) & (streaks >= min_streak)
    n_bad = bad_mask.sum()
    if n_bad > 0:
        logger.warning("Flagging %d readings in zero-streaks >= %d hours (likely outages)", n_bad, min_streak)
        df.loc[bad_mask, target_col] = np.nan

    return df


def data_quality_report(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    report_rows = []
    for site_id in sorted(df["site_id"].unique()):
        site_df = df[df["site_id"] == site_id]
        n_buildings = site_df["building_id"].nunique()
        n_meters = site_df.groupby(["building_id", "meter"]).ngroups
        n_rows = len(site_df)
        pct_missing = 100 * site_df["meter_reading"].isna().sum() / max(n_rows, 1)
        pct_zero = 100 * (site_df["meter_reading"] == 0).sum() / max(n_rows, 1)
        mean_reading = site_df["meter_reading"].mean()
        max_reading = site_df["meter_reading"].max()

        report_rows.append({
            "site_id": site_id,
            "n_buildings": n_buildings,
            "n_meters": n_meters,
            "n_rows": n_rows,
            "pct_missing": round(pct_missing, 2),
            "pct_zero": round(pct_zero, 2),
            "mean_reading": round(float(mean_reading), 2) if pd.notna(mean_reading) else 0,
            "max_reading": round(float(max_reading), 2) if pd.notna(max_reading) else 0,
        })

    report = pd.DataFrame(report_rows)
    logger.info("Data quality report:\n%s", report.to_string(index=False))
    return report


def stream_train_chunks(
    cfg: AppConfig,
    meta: pd.DataFrame,
    weather: pd.DataFrame,
    n_chunks: int | None = None,
) -> Generator[pd.DataFrame, None, None]:
    path = cfg.paths.train_path()
    logger.info("Streaming %s in chunks of %d", path, cfg.pipeline.chunk_size)

    reader = pd.read_csv(
        path,
        dtype=TRAIN_DTYPES,
        parse_dates=["timestamp"],
        chunksize=cfg.pipeline.chunk_size,
    )

    chunks = itertools.islice(reader, n_chunks) if n_chunks else reader

    for i, chunk in enumerate(chunks):
        chunk = chunk.merge(meta, on="building_id", how="left")
        chunk = chunk.merge(weather, on=["site_id", "timestamp"], how="left")
        chunk = _clean_readings(chunk)
        logger.info(
            "Chunk %d: %d rows, %.1f MB",
            i, len(chunk), chunk.memory_usage(deep=True).sum() / 1e6,
        )
        yield chunk


def load_full_dataset(
    cfg: AppConfig, n_chunks: int | None = None, clean: bool = True
) -> pd.DataFrame:
    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)
    frames = list(stream_train_chunks(cfg, meta, weather, n_chunks=n_chunks))
    df = pd.concat(frames, ignore_index=True)

    if clean:
        df = remove_outliers(df)
        df = detect_zero_streaks(df)

    return df


def load_site_data(cfg: AppConfig, site_id: int) -> pd.DataFrame:
    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)
    site_buildings = set(meta.loc[meta["site_id"] == site_id, "building_id"])

    site_chunks = []
    for chunk in stream_train_chunks(cfg, meta, weather):
        site_rows = chunk[chunk["building_id"].isin(site_buildings)]
        if len(site_rows) > 0:
            site_chunks.append(site_rows)

    if not site_chunks:
        return pd.DataFrame()

    df = pd.concat(site_chunks, ignore_index=True)
    df = remove_outliers(df)
    df = detect_zero_streaks(df)
    return df
