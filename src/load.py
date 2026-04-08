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


def load_building_metadata(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.paths.meta_path()
    logger.info("Loading building metadata from %s", path)
    meta = pd.read_csv(path, dtype=META_DTYPES)
    meta["primary_use"] = meta["primary_use"].astype("category")
    meta["building_age"] = 2017 - meta["year_built"]
    meta["log_square_feet"] = np.log1p(meta["square_feet"]).astype(np.float32)
    return meta


def load_weather(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.paths.weather_path()
    logger.info("Loading weather data from %s", path)
    weather = pd.read_csv(path, dtype=WEATHER_DTYPES, parse_dates=["timestamp"])
    weather["air_temperature"] = weather["air_temperature"].ffill()
    weather["dew_temperature"] = weather["dew_temperature"].ffill()
    return weather


def _build_weather_lookup(weather: pd.DataFrame) -> dict[tuple[int, pd.Timestamp], dict]:
    """O(1) weather lookups keyed by (site_id, timestamp)."""
    lookup: dict[tuple[int, pd.Timestamp], dict] = {}
    cols = [c for c in weather.columns if c not in ("site_id", "timestamp")]
    for row in weather.itertuples(index=False):
        key = (row.site_id, row.timestamp)
        lookup[key] = {c: getattr(row, c) for c in cols}
    return lookup


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
        logger.info("Chunk %d: %d rows, %.1f MB", i, len(chunk), chunk.memory_usage(deep=True).sum() / 1e6)
        yield chunk


def _clean_readings(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["meter_reading"] >= 0
    n_neg = (~mask).sum()
    if n_neg > 0:
        logger.warning("Dropping %d negative meter readings", n_neg)
        df = df.loc[mask].copy()
    return df


def load_full_dataset(cfg: AppConfig, n_chunks: int | None = None) -> pd.DataFrame:
    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)
    frames = list(stream_train_chunks(cfg, meta, weather, n_chunks=n_chunks))
    return pd.concat(frames, ignore_index=True)


def load_site_data(cfg: AppConfig, site_id: int) -> pd.DataFrame:
    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)
    site_buildings = set(meta.loc[meta["site_id"] == site_id, "building_id"])

    site_chunks = []
    for chunk in stream_train_chunks(cfg, meta, weather):
        site_rows = chunk[chunk["building_id"].isin(site_buildings)]
        if len(site_rows) > 0:
            site_chunks.append(site_rows)

    return pd.concat(site_chunks, ignore_index=True) if site_chunks else pd.DataFrame()
