from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METER_NAMES = {0: "Electricity", 1: "Chilled Water", 2: "Steam", 3: "Hot Water"}
COLORS = {"Electricity": "#2196F3", "Chilled Water": "#4CAF50", "Steam": "#FF5722", "Hot Water": "#FF9800"}


def run_eda(df: pd.DataFrame, meta: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running EDA on %d rows", len(df))
    summary = {}

    summary["dataset"] = _dataset_summary(df, meta, output_dir)
    summary["meter_distributions"] = _meter_distributions(df, output_dir)
    summary["temporal_patterns"] = _temporal_patterns(df, output_dir)
    summary["missing_data"] = _missing_data_analysis(df, meta, output_dir)
    summary["weather_correlation"] = _weather_correlation(df, output_dir)
    summary["site_comparison"] = _site_comparison(df, meta, output_dir)

    _save_summary(summary, output_dir)
    logger.info("EDA complete. Outputs in %s", output_dir)
    return summary


def _dataset_summary(df: pd.DataFrame, meta: pd.DataFrame, output_dir: Path) -> dict:
    stats = {
        "total_rows": len(df),
        "n_buildings": df["building_id"].nunique(),
        "n_sites": df["site_id"].nunique() if "site_id" in df.columns else 0,
        "n_meters": df.groupby(["building_id", "meter"]).ngroups,
        "date_range_start": str(df["timestamp"].min()),
        "date_range_end": str(df["timestamp"].max()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }

    meter_counts = df.groupby("meter")["building_id"].nunique()
    stats["meters_per_type"] = {METER_NAMES.get(k, str(k)): int(v) for k, v in meter_counts.items()}

    if "primary_use" in meta.columns:
        use_counts = meta["primary_use"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        use_counts.plot.barh(ax=ax, color="#2196F3")
        ax.set_xlabel("Number of Buildings")
        ax.set_title("Buildings by Primary Use")
        plt.tight_layout()
        fig.savefig(output_dir / "building_use_distribution.png", dpi=150)
        plt.close(fig)

    return stats


def _meter_distributions(df: pd.DataFrame, output_dir: Path) -> dict:
    stats = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for meter_code in range(4):
        name = METER_NAMES.get(meter_code, str(meter_code))
        subset = df[df["meter"] == meter_code]["meter_reading"]

        if len(subset) == 0:
            stats[name] = {"count": 0}
            continue

        stats[name] = {
            "count": int(len(subset)),
            "mean": round(float(subset.mean()), 2),
            "median": round(float(subset.median()), 2),
            "std": round(float(subset.std()), 2),
            "max": round(float(subset.max()), 2),
            "pct_zero": round(100 * (subset == 0).mean(), 2),
        }

        ax = axes[meter_code]
        clipped = subset.clip(upper=subset.quantile(0.99))
        ax.hist(clipped, bins=80, color=COLORS.get(name, "#999"), alpha=0.8, edgecolor="none")
        ax.set_title(f"{name} (n={len(subset):,})")
        ax.set_xlabel("kWh")
        ax.set_ylabel("Frequency")

    plt.suptitle("Meter Reading Distributions (clipped at 99th pctl)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "meter_distributions.png", dpi=150)
    plt.close(fig)

    return stats


def _temporal_patterns(df: pd.DataFrame, output_dir: Path) -> dict:
    ts = pd.DatetimeIndex(df["timestamp"])

    hourly = df.groupby(ts.hour)["meter_reading"].mean()
    daily = df.groupby(ts.dayofweek)["meter_reading"].mean()
    monthly = df.groupby(ts.month)["meter_reading"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(hourly.index, hourly.values, color="#2196F3")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Mean Reading")
    axes[0].set_title("Hourly Pattern")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[1].bar(range(7), daily.values, color="#4CAF50", tick_label=day_names)
    axes[1].set_ylabel("Mean Reading")
    axes[1].set_title("Weekly Pattern")

    axes[2].bar(monthly.index, monthly.values, color="#FF5722")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Mean Reading")
    axes[2].set_title("Monthly Pattern")

    plt.suptitle("Temporal Consumption Patterns", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "temporal_patterns.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for meter_code in range(4):
        name = METER_NAMES.get(meter_code, str(meter_code))
        subset = df[df["meter"] == meter_code]
        if len(subset) == 0:
            continue
        hourly_meter = subset.groupby(pd.DatetimeIndex(subset["timestamp"]).hour)["meter_reading"].mean()
        axes[meter_code].bar(hourly_meter.index, hourly_meter.values, color=COLORS.get(name, "#999"))
        axes[meter_code].set_title(f"{name}")
        axes[meter_code].set_xlabel("Hour")

    plt.suptitle("Hourly Pattern by Meter Type", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "hourly_by_meter.png", dpi=150)
    plt.close(fig)

    return {
        "peak_hour": int(hourly.idxmax()),
        "trough_hour": int(hourly.idxmin()),
        "weekday_vs_weekend_ratio": round(float(daily.iloc[:5].mean() / max(daily.iloc[5:].mean(), 1e-6)), 2),
    }


def _missing_data_analysis(df: pd.DataFrame, meta: pd.DataFrame, output_dir: Path) -> dict:
    cols_to_check = ["air_temperature", "dew_temperature", "cloud_coverage",
                     "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]
    present_cols = [c for c in cols_to_check if c in df.columns]

    pct_missing = {}
    for col in present_cols:
        pct = float(df[col].isna().mean() * 100)
        pct_missing[col] = round(pct, 2)

    if present_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(list(pct_missing.keys()), list(pct_missing.values()), color="#FF9800")
        ax.set_xlabel("% Missing")
        ax.set_title("Missing Data by Weather Column")
        for bar, val in zip(bars, pct_missing.values()):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        fig.savefig(output_dir / "missing_data.png", dpi=150)
        plt.close(fig)

    zero_pct = float((df["meter_reading"] == 0).mean() * 100)
    return {
        "weather_missing_pct": pct_missing,
        "meter_reading_zero_pct": round(zero_pct, 2),
        "meter_reading_na_pct": round(float(df["meter_reading"].isna().mean() * 100), 2),
    }


def _weather_correlation(df: pd.DataFrame, output_dir: Path) -> dict:
    if "air_temperature" not in df.columns:
        return {}

    sample = df.dropna(subset=["air_temperature", "meter_reading"])
    if len(sample) > 200_000:
        sample = sample.sample(200_000, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for meter_code in range(4):
        name = METER_NAMES.get(meter_code, str(meter_code))
        subset = sample[sample["meter"] == meter_code]
        if len(subset) < 100:
            continue
        binned = subset.groupby(pd.cut(subset["air_temperature"], 20))["meter_reading"].mean()
        axes[0].plot(
            [interval.mid for interval in binned.index],
            binned.values,
            label=name, color=COLORS.get(name, "#999"), linewidth=2,
        )

    axes[0].set_xlabel("Air Temperature (°C)")
    axes[0].set_ylabel("Mean Consumption")
    axes[0].set_title("Consumption vs Temperature")
    axes[0].legend()

    if "dew_temperature" in sample.columns:
        corr_cols = ["meter_reading", "air_temperature", "dew_temperature"]
        if "wind_speed" in sample.columns:
            corr_cols.append("wind_speed")
        corr = sample[corr_cols].corr()
        im = axes[1].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_xticks(range(len(corr_cols)))
        axes[1].set_yticks(range(len(corr_cols)))
        axes[1].set_xticklabels([c.replace("_", "\n") for c in corr_cols], fontsize=8)
        axes[1].set_yticklabels([c.replace("_", "\n") for c in corr_cols], fontsize=8)
        axes[1].set_title("Correlation Matrix")
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                axes[1].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "weather_correlation.png", dpi=150)
    plt.close(fig)

    return {
        "temp_meter_corr": round(float(sample[["air_temperature", "meter_reading"]].corr().iloc[0, 1]), 3),
    }


def _site_comparison(df: pd.DataFrame, meta: pd.DataFrame, output_dir: Path) -> dict:
    if "site_id" not in df.columns:
        return {}

    site_stats = df.groupby("site_id").agg(
        n_buildings=("building_id", "nunique"),
        n_rows=("meter_reading", "count"),
        mean_reading=("meter_reading", "mean"),
        median_reading=("meter_reading", "median"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(site_stats["site_id"], site_stats["mean_reading"], color="#2196F3")
    axes[0].set_xlabel("Site ID")
    axes[0].set_ylabel("Mean Consumption")
    axes[0].set_title("Mean Consumption by Site")

    axes[1].bar(site_stats["site_id"], site_stats["n_buildings"], color="#4CAF50")
    axes[1].set_xlabel("Site ID")
    axes[1].set_ylabel("Buildings")
    axes[1].set_title("Buildings per Site")

    plt.tight_layout()
    fig.savefig(output_dir / "site_comparison.png", dpi=150)
    plt.close(fig)

    return {
        "highest_consumption_site": int(site_stats.loc[site_stats["mean_reading"].idxmax(), "site_id"]),
        "most_buildings_site": int(site_stats.loc[site_stats["n_buildings"].idxmax(), "site_id"]),
    }


def _save_summary(summary: dict, output_dir: Path):
    import json
    out_path = output_dir / "eda_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("EDA summary saved to %s", out_path)


def plot_feature_importance(importance: dict, output_dir: str | Path, top_n: int = 20):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), values, color="#2196F3")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_n} Feature Importance (LightGBM)")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_predictions_vs_actual(actual: np.ndarray, predicted: np.ndarray, output_dir: str | Path, sample_n: int = 5000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(actual) > sample_n:
        idx = np.random.RandomState(42).choice(len(actual), sample_n, replace=False)
        actual = actual[idx]
        predicted = predicted[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    max_val = max(actual.max(), predicted.max())
    axes[0].scatter(actual, predicted, alpha=0.15, s=4, color="#2196F3")
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")

    residuals = actual - predicted
    axes[1].hist(residuals, bins=100, color="#FF5722", alpha=0.8, edgecolor="none")
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residual Distribution (mean={residuals.mean():.1f})")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    fig.savefig(output_dir / "predictions_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("baseline_mean_rmse", "Baseline\nMean"),
        ("baseline_lag24_rmse", "Baseline\nLag-24h"),
        ("xgboost_rmse", "XGBoost"),
        ("lightgbm_rmse", "LightGBM"),
    ]
    models = []
    rmses = []
    for key, label in candidates:
        if key in results:
            models.append(label)
            rmses.append(results[key])

    if not models:
        return

    colors = ["#FF9800", "#FF5722", "#9C27B0", "#2196F3"][:len(models)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(models, rmses, color=colors, width=0.5)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmses) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison")
    plt.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_benchmark_speedups(benchmarks_csv: str | Path, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_csv = Path(benchmarks_csv)

    if not benchmarks_csv.exists():
        return

    df = pd.read_csv(benchmarks_csv)
    components = df["component"].unique()

    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 4))
    if len(components) == 1:
        axes = [axes]

    for ax, comp in zip(axes, components):
        subset = df[df["component"] == comp]
        methods = subset["method"].values
        times = subset["time_seconds"].values
        ax.barh(methods, times, color="#2196F3")
        ax.set_xlabel("Time (seconds)")
        ax.set_title(comp)
        for i, (m, t) in enumerate(zip(methods, times)):
            ax.text(t + max(times) * 0.02, i, f"{t:.3f}s", va="center", fontsize=8)

    plt.suptitle("Optimization Benchmarks", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "benchmark_speedups.png", dpi=150)
    plt.close(fig)


def plot_parallel_speedup(parallel_csv: str | Path, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parallel_csv = Path(parallel_csv)

    if not parallel_csv.exists():
        return

    df = pd.read_csv(parallel_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(df["n_workers"], df["time_seconds"], "o-", color="#2196F3", linewidth=2)
    axes[0].set_xlabel("Workers")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Training Time vs Workers")

    axes[1].plot(df["n_workers"], df["speedup"], "o-", color="#4CAF50", linewidth=2)
    axes[1].plot(df["n_workers"], df["n_workers"], "--", color="#999", alpha=0.5, label="ideal")
    axes[1].set_xlabel("Workers")
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Parallel Speedup")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "parallel_speedup.png", dpi=150)
    plt.close(fig)


def plot_anomaly_distribution(val_df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(val_df["anomaly_score"].clip(-10, 10), bins=100, color="#FF5722", alpha=0.8, edgecolor="none")
    axes[0].axvline(-3.5, color="red", linestyle="--", linewidth=1, label="threshold")
    axes[0].axvline(3.5, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Anomaly Score Distribution")
    axes[0].legend()

    meter_anomaly = val_df.groupby("meter")["is_anomaly"].mean() * 100
    meter_labels = [METER_NAMES.get(m, str(m)) for m in meter_anomaly.index]
    axes[1].bar(meter_labels, meter_anomaly.values, color=[COLORS.get(n, "#999") for n in meter_labels])
    axes[1].set_ylabel("Anomaly Rate (%)")
    axes[1].set_title("Anomaly Rate by Meter Type")

    plt.tight_layout()
    fig.savefig(output_dir / "anomaly_distribution.png", dpi=150)
    plt.close(fig)
