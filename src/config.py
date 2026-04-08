from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    data_dir: str = "data"
    train_file: str = "train.csv"
    building_meta_file: str = "building_metadata.csv"
    weather_file: str = "weather_train.csv"
    results_dir: str = "results"

    def train_path(self) -> Path:
        return Path(self.data_dir) / self.train_file

    def meta_path(self) -> Path:
        return Path(self.data_dir) / self.building_meta_file

    def weather_path(self) -> Path:
        return Path(self.data_dir) / self.weather_file

    def results_path(self) -> Path:
        p = Path(self.results_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(slots=True)
class PipelineConfig:
    chunk_size: int = 2_000_000
    seed: int = 42
    validation_months: int = 3
    target_col: str = "meter_reading"


@dataclass(slots=True)
class FeaturesConfig:
    lag_hours: list[int] = field(default_factory=lambda: [24, 168])
    rolling_windows: list[int] = field(default_factory=lambda: [24, 168])
    time_features: list[str] = field(
        default_factory=lambda: ["hour", "dayofweek", "month", "is_weekend"]
    )
    use_holidays: bool = True
    dtype: str = "float32"


@dataclass(slots=True)
class ModelConfig:
    lgbm_params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    })
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50


@dataclass(slots=True)
class AnomalyConfig:
    method: str = "modified_zscore"
    threshold: float = 3.5
    min_hours: int = 100


@dataclass(slots=True)
class ParallelConfig:
    n_workers: int = 4
    backend: str = "process"


@dataclass(slots=True)
class ProfilingConfig:
    enabled: bool = True
    output_dir: str = "results/profiling"


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)


def _build_dataclass(cls, raw: dict | None):
    if raw is None:
        return cls()
    filtered = {k: v for k, v in raw.items() if k in {f.name for f in cls.__dataclass_fields__.values()}}
    return cls(**filtered)


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        return AppConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(
        paths=_build_dataclass(PathsConfig, raw.get("paths")),
        pipeline=_build_dataclass(PipelineConfig, raw.get("pipeline")),
        features=_build_dataclass(FeaturesConfig, raw.get("features")),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        anomaly=_build_dataclass(AnomalyConfig, raw.get("anomaly")),
        parallel=_build_dataclass(ParallelConfig, raw.get("parallel")),
        profiling=_build_dataclass(ProfilingConfig, raw.get("profiling")),
    )
