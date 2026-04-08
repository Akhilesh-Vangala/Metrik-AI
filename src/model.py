from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import AppConfig
from src.features import get_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    rmse: float
    cv_rmse: float
    mae: float
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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mean_actual = float(np.mean(y_true))
    cv_rmse = rmse / mean_actual if mean_actual > 0 else float("inf")
    return {"rmse": rmse, "mae": mae, "cv_rmse": cv_rmse, "mean_actual": mean_actual}


def train_baseline_mean(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "meter_reading",
) -> ModelResult:
    t0 = time.perf_counter()

    meter_means = train.groupby(["building_id", "meter"])[target_col].mean().to_dict()
    global_mean = float(train[target_col].mean())
    preds = np.array(
        [meter_means.get((row.building_id, row.meter), global_mean)
         for row in val[["building_id", "meter"]].itertuples(index=False)],
        dtype=np.float32,
    )

    elapsed = time.perf_counter() - t0
    metrics = _compute_metrics(val[target_col].values, preds)
    logger.info("Baseline mean: RMSE=%.4f, CV-RMSE=%.4f, time=%.2fs", metrics["rmse"], metrics["cv_rmse"], elapsed)
    return ModelResult("baseline_mean", preds, metrics["rmse"], metrics["cv_rmse"], metrics["mae"], elapsed, {})


def train_baseline_lag(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "meter_reading",
) -> ModelResult:
    t0 = time.perf_counter()

    if "lag_24h" in val.columns:
        preds = val["lag_24h"].fillna(val[target_col].mean()).values.astype(np.float32)
    else:
        preds = np.full(len(val), train[target_col].mean(), dtype=np.float32)

    elapsed = time.perf_counter() - t0
    metrics = _compute_metrics(val[target_col].values, preds)
    logger.info("Baseline lag24: RMSE=%.4f, CV-RMSE=%.4f, time=%.2fs", metrics["rmse"], metrics["cv_rmse"], elapsed)
    return ModelResult("baseline_lag24", preds, metrics["rmse"], metrics["cv_rmse"], metrics["mae"], elapsed, {})


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

    logger.info("Training LightGBM: %d features, %d train rows, %d val rows", len(feat_cols), len(X_train), len(X_val))
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

    metrics = _compute_metrics(y_val, preds)
    importance = dict(zip(feat_cols, model.feature_importance(importance_type="gain").tolist()))

    logger.info(
        "LightGBM: RMSE=%.4f, CV-RMSE=%.4f, MAE=%.4f, time=%.2fs, best_iter=%d",
        metrics["rmse"], metrics["cv_rmse"], metrics["mae"], elapsed, model.best_iteration,
    )

    return ModelResult(
        "lightgbm", preds, metrics["rmse"], metrics["cv_rmse"], metrics["mae"], elapsed,
        {"model": model, "feature_importance": importance, "n_features": len(feat_cols),
         "best_iteration": model.best_iteration, "features_used": feat_cols},
    )


def train_site_model(
    site_id: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: AppConfig,
) -> dict:
    site_train = train[train["site_id"] == site_id]
    site_val = val[val["site_id"] == site_id]

    if len(site_train) < 100 or len(site_val) < 10:
        logger.warning("Site %d: insufficient data (train=%d, val=%d), skipping", site_id, len(site_train), len(site_val))
        return {"site_id": site_id, "status": "skipped", "rmse": None}

    result = train_lightgbm(site_train, site_val, cfg)
    return {
        "site_id": site_id,
        "status": "ok",
        "rmse": result.rmse,
        "cv_rmse": result.cv_rmse,
        "mae": result.mae,
        "train_time": result.train_time,
        "n_train": len(site_train),
        "n_val": len(site_val),
        "predictions": result.predictions,
    }


def save_model(model: lgb.Booster, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def save_results(results: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in results.items() if not isinstance(v, (np.ndarray, lgb.Booster))}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


def save_predictions(
    val_df: pd.DataFrame,
    predictions: np.ndarray,
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = val_df[["building_id", "meter", "timestamp", "meter_reading"]].copy()
    out["predicted"] = predictions
    out["residual"] = out["meter_reading"] - out["predicted"]
    out.to_csv(path, index=False)
    logger.info("Predictions saved to %s (%d rows)", path, len(out))


def compute_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (actual - predicted).astype(np.float32)
