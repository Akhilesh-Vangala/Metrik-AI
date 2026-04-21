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
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

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
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    cutoff = max_date - pd.DateOffset(months=validation_months)

    train = df[df["timestamp"] < cutoff].copy()
    val = df[df["timestamp"] >= cutoff].copy()

    if len(train) == 0:
        span_months = (max_date - min_date).days / 30
        raise ValueError(
            f"Train set is empty after time-based split. "
            f"Data spans only {span_months:.1f} months "
            f"({min_date.date()} to {max_date.date()}) but "
            f"validation_months={validation_months} consumes all of it. "
            f"Load more data: use --n-chunks 6 or more (full run: omit --n-chunks)."
        )

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


def train_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: AppConfig,
) -> ModelResult:
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost not installed: pip install xgboost")

    target = cfg.pipeline.target_col
    feat_cols = get_feature_columns(train)

    X_train = train[feat_cols].values.astype(np.float32)
    y_train = train[target].values.astype(np.float32)
    X_val = val[feat_cols].values.astype(np.float32)
    y_val = val[target].values.astype(np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_cols)

    params = {
        "objective": "reg:squarederror",
        "learning_rate": cfg.model.lgbm_params.get("learning_rate", 0.05),
        "max_depth": 6,
        "subsample": cfg.model.lgbm_params.get("bagging_fraction", 0.8),
        "colsample_bytree": cfg.model.lgbm_params.get("feature_fraction", 0.8),
        "min_child_weight": 20,
        "tree_method": "hist",
        "verbosity": 0,
        "seed": cfg.pipeline.seed,
    }

    logger.info("Training XGBoost: %d features, %d train rows, %d val rows", len(feat_cols), len(X_train), len(X_val))
    t0 = time.perf_counter()

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.model.num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=cfg.model.early_stopping_rounds,
        verbose_eval=False,
    )

    elapsed = time.perf_counter() - t0
    preds = np.clip(model.predict(dval), 0, None).astype(np.float32)
    metrics = _compute_metrics(y_val, preds)

    logger.info(
        "XGBoost: RMSE=%.4f, CV-RMSE=%.4f, MAE=%.4f, time=%.2fs, best_iter=%d",
        metrics["rmse"], metrics["cv_rmse"], metrics["mae"], elapsed, model.best_iteration,
    )

    return ModelResult(
        "xgboost", preds, metrics["rmse"], metrics["cv_rmse"], metrics["mae"], elapsed,
        {"best_iteration": model.best_iteration, "n_features": len(feat_cols)},
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


def optimize_learning_rate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: AppConfig,
    lr_bounds: tuple[float, float] = (0.005, 0.3),
) -> float:
    """Find optimal LightGBM learning rate via scipy.optimize.minimize_scalar."""
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

    def objective(lr: float) -> float:
        params = {**cfg.model.lgbm_params, "learning_rate": float(lr), "verbose": -1}
        model = lgb.train(
            params, dtrain, num_boost_round=200,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    result = minimize_scalar(objective, bounds=lr_bounds, method="bounded")
    optimal_lr = float(result.x)
    logger.info("Optimal learning rate: %.5f (RMSE=%.4f)", optimal_lr, result.fun)
    return optimal_lr


def compute_residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (actual - predicted).astype(np.float32)
