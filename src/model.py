from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.config import AppConfig
from src.features import get_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    rmse: float
    cv_rmse: float
    train_time: float
    metadata: dict[str, Any]


def time_based_split(
    df: pd.DataFrame,
    validation_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["timestamp"].max()
    cutoff = max_date - pd.DateOffset(months=validation_months)

    train = df[df["timestamp"] < cutoff].copy()
    val = df[df["timestamp"] >= cutoff].copy()

    logger.info(
        "Split: train %d rows (to %s), val %d rows (from %s)",
        len(train), cutoff.date(), len(val), cutoff.date(),
    )
    return train, val


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_actual = np.mean(y_true)
    cv_rmse = rmse / mean_actual if mean_actual > 0 else float("inf")
    return float(rmse), float(cv_rmse)


def train_baseline_mean(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "meter_reading",
) -> ModelResult:
    t0 = time.perf_counter()

    meter_means = train.groupby(["building_id", "meter"])[target_col].mean().to_dict()
    global_mean = train[target_col].mean()
    preds = np.array(
        [meter_means.get((row.building_id, row.meter), global_mean)
         for row in val[["building_id", "meter"]].itertuples(index=False)],
        dtype=np.float32,
    )

    elapsed = time.perf_counter() - t0
    rmse, cv_rmse = _compute_metrics(val[target_col].values, preds)

    logger.info("Baseline mean: RMSE=%.4f, CV-RMSE=%.4f, time=%.2fs", rmse, cv_rmse, elapsed)
    return ModelResult("baseline_mean", preds, rmse, cv_rmse, elapsed, {})


def train_baseline_lag(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "meter_reading",
) -> ModelResult:
    t0 = time.perf_counter()

    if "lag_24h" in val.columns:
        preds = val["lag_24h"].fillna(val[target_col].mean()).values
    else:
        preds = np.full(len(val), train[target_col].mean(), dtype=np.float32)

    elapsed = time.perf_counter() - t0
    rmse, cv_rmse = _compute_metrics(val[target_col].values, preds)

    logger.info("Baseline lag24: RMSE=%.4f, CV-RMSE=%.4f, time=%.2fs", rmse, cv_rmse, elapsed)
    return ModelResult("baseline_lag24", preds, rmse, cv_rmse, elapsed, {})


def train_lightgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: AppConfig,
) -> ModelResult:
    target = cfg.pipeline.target_col
    feat_cols = get_feature_columns(train)
    cat_cols = [c for c in ["primary_use_code", "site_id", "meter"] if c in feat_cols]

    X_train = train[feat_cols].copy()
    y_train = train[target].values
    X_val = val[feat_cols].copy()
    y_val = val[target].values

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)

    logger.info("Training LightGBM with %d features on %d rows", len(feat_cols), len(X_train))
    t0 = time.perf_counter()

    callbacks = [
        lgb.early_stopping(cfg.model.early_stopping_rounds),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        cfg.model.lgbm_params,
        dtrain,
        num_boost_round=cfg.model.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    elapsed = time.perf_counter() - t0

    preds = model.predict(X_val)
    preds = np.clip(preds, 0, None).astype(np.float32)

    rmse, cv_rmse = _compute_metrics(y_val, preds)
    logger.info("LightGBM: RMSE=%.4f, CV-RMSE=%.4f, time=%.2fs", rmse, cv_rmse, elapsed)

    importance = dict(zip(feat_cols, model.feature_importance(importance_type="gain")))

    return ModelResult(
        "lightgbm", preds, rmse, cv_rmse, elapsed,
        {"model": model, "feature_importance": importance, "n_features": len(feat_cols)},
    )


def compute_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (actual - predicted).astype(np.float32)
