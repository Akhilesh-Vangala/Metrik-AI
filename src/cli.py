from __future__ import annotations

import logging
import multiprocessing
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.config import load_config
from src.load import (
    load_building_metadata, load_weather, stream_train_chunks,
    load_full_dataset, remove_outliers, detect_zero_streaks, data_quality_report,
)
from src.features import build_features, build_features_naive, get_feature_columns
from src.model import (
    time_based_split, train_baseline_mean, train_baseline_lag, train_lightgbm,
    train_xgboost, train_site_model, compute_residuals, save_model, save_results,
    save_predictions, XGB_AVAILABLE,
)
from src.anomaly import detect_anomalies, aggregate_anomalies, summarize_anomalies_by_site
from src.decision import build_audit_list, export_audit_list
from src.benchmark import BenchmarkSuite, profile_function, measure_memory
from src.eda import (
    run_eda, plot_feature_importance, plot_predictions_vs_actual,
    plot_anomaly_distribution, plot_model_comparison,
    plot_benchmark_speedups, plot_parallel_speedup,
)


_BENCH_TRAIN = None
_BENCH_VAL = None
_BENCH_CFG = None


def _site_bench_worker(site_id: int) -> dict:
    return train_site_model(site_id, _BENCH_TRAIN, _BENCH_VAL, _BENCH_CFG)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--config", "config_path", default="config/config.yaml", help="Path to config file")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def cli(ctx, config_path: str, verbose: bool):
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config_path)


@cli.command()
@click.option("--n-chunks", type=int, default=None, help="Limit chunks for dev mode")
@click.option("--save-model-path", default="results/model.lgb", help="Where to save the trained model")
@click.pass_context
def run(ctx, n_chunks: int | None, save_model_path: str):
    """Full pipeline: load -> clean -> features -> model -> anomaly -> audit."""
    cfg = ctx.obj["cfg"]
    t_start = time.perf_counter()
    results_dir = cfg.paths.results_path()
    log = logging.getLogger("pipeline")

    log.info("=== Metrik AI Pipeline ===")

    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)

    log.info("--- Stage 1: Data Loading ---")
    frames = []
    for chunk in stream_train_chunks(cfg, meta, weather, n_chunks=n_chunks):
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    del frames

    log.info("--- Stage 2: Data Cleaning ---")
    df = remove_outliers(df)
    df = detect_zero_streaks(df)
    quality = data_quality_report(df, meta)
    quality.to_csv(results_dir / "data_quality.csv", index=False)

    log.info("--- Stage 3: Feature Engineering ---")
    df = build_features(df, cfg.features)
    feat_cols = get_feature_columns(df)
    df = df.dropna(subset=feat_cols + [cfg.pipeline.target_col])

    log.info("--- Stage 4: Train/Val Split ---")
    train_df, val_df = time_based_split(df, cfg.pipeline.validation_months)

    log.info("--- Stage 5: Model Training ---")
    baseline_result = train_baseline_mean(train_df, val_df, cfg.pipeline.target_col)
    lag_result = train_baseline_lag(train_df, val_df, cfg.pipeline.target_col)
    lgbm_result = train_lightgbm(train_df, val_df, cfg)

    if lgbm_result.metadata.get("model"):
        save_model(lgbm_result.metadata["model"], save_model_path)

    save_predictions(val_df, lgbm_result.predictions, results_dir / "predictions.csv")

    log.info("--- Stage 6: Anomaly Detection ---")
    residuals = compute_residuals(val_df[cfg.pipeline.target_col].values, lgbm_result.predictions)
    val_df = val_df.copy()
    val_df["residual"] = residuals
    val_df = detect_anomalies(val_df, cfg=cfg.anomaly)
    anomaly_summary = aggregate_anomalies(val_df)
    anomaly_summary.to_csv(results_dir / "anomaly_summary.csv", index=False)

    site_summary = summarize_anomalies_by_site(val_df)
    for sid, info in site_summary.items():
        log.info("Site %d: %d anomalies, mean severity=%.2f", sid, info["n_anomalies"], info["mean_severity"])

    log.info("--- Stage 7: Decision Support ---")
    audit = build_audit_list(anomaly_summary, meta, min_hours=cfg.anomaly.min_hours)
    export_audit_list(audit, results_dir / "audit_list.csv")

    elapsed = time.perf_counter() - t_start

    rmse_improvement = (1 - lgbm_result.rmse / baseline_result.rmse) * 100

    all_results = {
        "rows_processed": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "n_features": len(feat_cols),
        "baseline_mean_rmse": baseline_result.rmse,
        "baseline_lag24_rmse": lag_result.rmse,
        "lightgbm_rmse": lgbm_result.rmse,
        "lightgbm_cv_rmse": lgbm_result.cv_rmse,
        "lightgbm_mae": lgbm_result.mae,
        "rmse_improvement_pct": round(rmse_improvement, 2),
        "anomalies_flagged": int(val_df["is_anomaly"].sum()),
        "anomaly_rate_pct": round(100 * val_df["is_anomaly"].mean(), 2),
        "audit_list_entries": len(audit),
        "total_time_seconds": round(elapsed, 2),
        "feature_importance": lgbm_result.metadata.get("feature_importance", {}),
    }
    save_results(all_results, results_dir / "pipeline_results.json")

    log.info("--- Stage 8: Generating Visualizations ---")
    plots_dir = results_dir / "plots"
    if lgbm_result.metadata.get("feature_importance"):
        plot_feature_importance(lgbm_result.metadata["feature_importance"], plots_dir)
    plot_predictions_vs_actual(val_df[cfg.pipeline.target_col].values, lgbm_result.predictions, plots_dir)
    plot_anomaly_distribution(val_df, plots_dir)
    plot_model_comparison(all_results, plots_dir)

    print(f"\n{'='*60}")
    print(f"  Metrik AI Pipeline Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Rows processed:      {len(df):,}")
    print(f"  Features used:       {len(feat_cols)}")
    print(f"  Baseline mean RMSE:  {baseline_result.rmse:.4f}")
    print(f"  Baseline lag24 RMSE: {lag_result.rmse:.4f}")
    print(f"  LightGBM RMSE:       {lgbm_result.rmse:.4f}")
    print(f"  LightGBM MAE:        {lgbm_result.mae:.4f}")
    print(f"  RMSE improvement:    {rmse_improvement:.1f}%")
    print(f"  Anomalies flagged:   {val_df['is_anomaly'].sum():,}")
    print(f"  Audit list entries:  {len(audit)}")
    print(f"  Results saved to:    {results_dir}/")
    print(f"{'='*60}\n")


@cli.command()
@click.option("--n-chunks", type=int, default=2)
@click.pass_context
def benchmark(ctx, n_chunks: int):
    """Run comprehensive benchmarks across all optimization techniques."""
    from src.numba_ops import modified_zscore_numba, rolling_mean_numba, warmup as numba_warmup
    from src.anomaly import modified_zscore_naive, modified_zscore_vectorized

    cfg = ctx.obj["cfg"]
    suite = BenchmarkSuite()

    print("\n=== Anomaly Scoring Benchmarks ===")
    for size in [100_000, 500_000, 1_000_000]:
        data = np.random.randn(size).astype(np.float64)

        suite.time_function("anomaly_scoring", "python_loop", modified_zscore_naive, data, input_size=size)
        suite.time_function("anomaly_scoring", "numpy_vectorized", modified_zscore_vectorized, data, input_size=size)

        numba_warmup()
        suite.time_function("anomaly_scoring", "numba_jit", modified_zscore_numba, data, input_size=size)

    try:
        from src.cython_kernels import modified_zscore_cython
        for size in [100_000, 500_000, 1_000_000]:
            data = np.random.randn(size).astype(np.float64)
            suite.time_function("anomaly_scoring", "cython", modified_zscore_cython, data, input_size=size)
    except ImportError:
        print("  (Cython not compiled — run: python setup.py build_ext --inplace)")

    try:
        from src.gpu_ops import modified_zscore_gpu, GPU_AVAILABLE, warmup as gpu_warmup
        if GPU_AVAILABLE:
            try:
                gpu_warmup()
                for size in [100_000, 500_000, 1_000_000]:
                    data = np.random.randn(size).astype(np.float64)
                    suite.time_function("anomaly_scoring", "cupy_gpu", modified_zscore_gpu, data, input_size=size)
            except Exception as exc:
                print(f"  (GPU benchmark aborted at runtime: {type(exc).__name__}: {exc})")
        else:
            print("  (GPU not available — CuPy benchmarks skipped)")
    except ImportError:
        print("  (CuPy not installed — GPU benchmarks skipped)")

    print("\n=== Feature Engineering Benchmarks ===")
    from src.features import build_features, build_features_naive

    rng = np.random.RandomState(cfg.pipeline.seed)

    def _make_sample(n):
        return pd.DataFrame({
            "building_id": rng.randint(0, 50, n).astype(np.int16),
            "meter": rng.randint(0, 4, n).astype(np.int8),
            "timestamp": pd.date_range("2016-01-01", periods=n, freq="h"),
            "meter_reading": rng.exponential(200, n).astype(np.float32),
            "site_id": rng.randint(0, 5, n).astype(np.int8),
            "primary_use": pd.Categorical(rng.choice(["Education", "Office", "Lodging"], n)),
            "square_feet": rng.uniform(5000, 100000, n).astype(np.float32),
            "log_square_feet": np.log1p(rng.uniform(5000, 100000, n)).astype(np.float32),
            "year_built": rng.uniform(1960, 2015, n).astype(np.float32),
            "building_age": rng.uniform(2, 57, n).astype(np.float32),
            "air_temperature": rng.uniform(-5, 40, n).astype(np.float32),
            "dew_temperature": rng.uniform(-10, 30, n).astype(np.float32),
            "wind_speed": rng.uniform(0, 15, n).astype(np.float32),
            "cloud_coverage": rng.uniform(0, 9, n).astype(np.float32),
        })

    for feat_size in [5_000, 10_000, 50_000]:
        sample = _make_sample(feat_size)
        suite.time_function("feature_engineering", "naive_iterrows", build_features_naive, sample.copy(), cfg.features, input_size=feat_size)
        suite.time_function("feature_engineering", "vectorized", build_features, sample.copy(), cfg.features, input_size=feat_size)

    print("\n=== Rolling Window Benchmarks ===")
    from src.numba_ops import rolling_mean_numba
    for size in [100_000, 500_000]:
        data = rng.randn(size).astype(np.float64)

        def _pandas_rolling(d):
            return pd.Series(d).rolling(24, min_periods=6).mean().values

        suite.time_function("rolling_mean", "pandas", _pandas_rolling, data, input_size=size)
        suite.time_function("rolling_mean", "numba_jit", rolling_mean_numba, data, 24, input_size=size)

    print("\n=== Memory Benchmarks ===")
    _, mem_full = measure_memory(lambda: pd.DataFrame({"x": np.random.randn(2_000_000).astype(np.float64)}))
    _, mem_f32 = measure_memory(lambda: pd.DataFrame({"x": np.random.randn(2_000_000).astype(np.float32)}))
    print(f"  float64 (2M rows): {mem_full:.1f} MB")
    print(f"  float32 (2M rows): {mem_f32:.1f} MB")
    print(f"  Memory reduction:  {(1 - mem_f32 / mem_full) * 100:.0f}%")

    suite.print_summary()

    results_path = cfg.paths.results_path() / "benchmarks.csv"
    pd.DataFrame(suite.summary_table()).to_csv(results_path, index=False)
    plot_benchmark_speedups(results_path, cfg.paths.results_path() / "plots")
    print(f"\nBenchmarks saved to {results_path}")


@cli.command()
@click.option("--n-workers-list", default="1,2,4,8,16", help="Comma-separated worker counts")
@click.option("--n-chunks", type=int, default=2)
@click.pass_context
def parallel_benchmark(ctx, n_workers_list: str, n_chunks: int):
    """Measure parallel speedup across different worker counts."""
    global _BENCH_TRAIN, _BENCH_VAL, _BENCH_CFG

    cfg = ctx.obj["cfg"]
    worker_counts = [int(x.strip()) for x in n_workers_list.split(",")]

    log = logging.getLogger("parallel")
    log.info("Loading data for parallel benchmark...")

    df = load_full_dataset(cfg, n_chunks=n_chunks)
    df = build_features(df, cfg.features)
    feat_cols = get_feature_columns(df)
    df = df.dropna(subset=feat_cols + [cfg.pipeline.target_col])
    train_df, val_df = time_based_split(df, cfg.pipeline.validation_months)

    _BENCH_TRAIN = train_df
    _BENCH_VAL = val_df
    _BENCH_CFG = cfg

    site_ids = sorted(train_df["site_id"].unique().tolist())
    log.info("Sites to train: %s", site_ids)

    print(f"\n=== Parallel Training Speedup (sites={len(site_ids)}) ===")
    timings = {}

    t0 = time.perf_counter()
    for sid in site_ids:
        _site_bench_worker(sid)
    timings[1] = time.perf_counter() - t0
    print(f"  Sequential (1 worker): {timings[1]:.2f}s")

    ctx_fork = multiprocessing.get_context("fork")
    for n in worker_counts:
        if n <= 1:
            continue
        t0 = time.perf_counter()
        with ctx_fork.Pool(processes=n) as pool:
            pool.map(_site_bench_worker, site_ids)
        timings[n] = time.perf_counter() - t0
        speedup = timings[1] / timings[n] if timings[n] > 0 else 0
        print(f"  {n} workers: {timings[n]:.2f}s (speedup: {speedup:.1f}x)")

    results_path = cfg.paths.results_path() / "parallel_benchmark.csv"
    pd.DataFrame([
        {"n_workers": k, "time_seconds": round(v, 3), "speedup": round(timings[1] / v, 2) if v > 0 else 0}
        for k, v in sorted(timings.items())
    ]).to_csv(results_path, index=False)
    plot_parallel_speedup(results_path, cfg.paths.results_path() / "plots")
    print(f"\nParallel benchmarks saved to {results_path}")


@cli.command()
@click.pass_context
def profile(ctx):
    """Profile the pipeline with cProfile and save results."""
    cfg = ctx.obj["cfg"]
    output_dir = Path(cfg.profiling.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def pipeline_subset():
        df = load_full_dataset(cfg, n_chunks=2)
        build_features(df, cfg.features)

    report = profile_function(pipeline_subset, output_path=str(output_dir / "pipeline.prof"))
    print(report)

    _, peak_mb = measure_memory(pipeline_subset)
    print(f"\nPeak memory for 2-chunk pipeline: {peak_mb:.1f} MB")


@cli.command()
@click.pass_context
def spark(ctx):
    """Run PySpark pipeline on the full dataset."""
    from src.spark_pipeline import run_spark_pipeline
    cfg = ctx.obj["cfg"]
    result = run_spark_pipeline(cfg.paths.data_dir)
    print(f"Spark pipeline: {result['rows']:,} rows in {result['elapsed_seconds']:.1f}s")


@cli.command()
@click.option("--n-chunks", type=int, default=2)
@click.pass_context
def quality(ctx, n_chunks: int):
    """Generate data quality report."""
    cfg = ctx.obj["cfg"]
    meta = load_building_metadata(cfg)
    df = load_full_dataset(cfg, n_chunks=n_chunks)
    report = data_quality_report(df, meta)
    out = cfg.paths.results_path() / "data_quality.csv"
    report.to_csv(out, index=False)
    print(f"Quality report saved to {out}")
    print(report.to_string(index=False))


@cli.command()
@click.option("--n-chunks", type=int, default=3, help="Number of chunks to analyze")
@click.pass_context
def eda(ctx, n_chunks: int):
    """Run exploratory data analysis and generate plots."""
    cfg = ctx.obj["cfg"]
    meta = load_building_metadata(cfg)
    df = load_full_dataset(cfg, n_chunks=n_chunks)
    output_dir = cfg.paths.results_path() / "eda"
    summary = run_eda(df, meta, output_dir)

    print(f"\n{'='*50}")
    print("  EDA Summary")
    print(f"{'='*50}")
    ds = summary.get("dataset", {})
    print(f"  Rows:       {ds.get('total_rows', 0):,}")
    print(f"  Buildings:  {ds.get('n_buildings', 0)}")
    print(f"  Sites:      {ds.get('n_sites', 0)}")
    print(f"  Date range: {ds.get('date_range_start', '?')} to {ds.get('date_range_end', '?')}")
    tp = summary.get("temporal_patterns", {})
    print(f"  Peak hour:  {tp.get('peak_hour', '?')}")
    print(f"  Weekday/Weekend ratio: {tp.get('weekday_vs_weekend_ratio', '?')}")
    md = summary.get("missing_data", {})
    print(f"  Zero readings: {md.get('meter_reading_zero_pct', '?')}%")
    print(f"  Plots saved to: {output_dir}/")
    print(f"{'='*50}\n")


@cli.command()
@click.option("--n-chunks", type=int, default=None)
@click.pass_context
def compare(ctx, n_chunks: int | None):
    """Train LightGBM and XGBoost side-by-side and report RMSE and speed."""
    if not XGB_AVAILABLE:
        print("xgboost not installed. Run: pip install xgboost")
        return

    cfg = ctx.obj["cfg"]
    log = logging.getLogger("compare")
    results_dir = cfg.paths.results_path()

    log.info("Loading data...")
    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)
    frames = []
    for chunk in stream_train_chunks(cfg, meta, weather, n_chunks=n_chunks):
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    df = remove_outliers(df)
    df = detect_zero_streaks(df)

    df = build_features(df, cfg.features)
    feat_cols = get_feature_columns(df)
    df = df.dropna(subset=feat_cols + [cfg.pipeline.target_col])
    train_df, val_df = time_based_split(df, cfg.pipeline.validation_months)

    log.info("Training baselines...")
    baseline_result = train_baseline_mean(train_df, val_df, cfg.pipeline.target_col)
    lag_result = train_baseline_lag(train_df, val_df, cfg.pipeline.target_col)

    log.info("Training LightGBM...")
    lgbm_result = train_lightgbm(train_df, val_df, cfg)

    log.info("Training XGBoost...")
    xgb_result = train_xgboost(train_df, val_df, cfg)

    speed_ratio = xgb_result.train_time / lgbm_result.train_time if lgbm_result.train_time > 0 else float("inf")

    comparison = {
        "baseline_mean_rmse": baseline_result.rmse,
        "baseline_lag24_rmse": lag_result.rmse,
        "xgboost_rmse": xgb_result.rmse,
        "xgboost_cv_rmse": xgb_result.cv_rmse,
        "xgboost_mae": xgb_result.mae,
        "xgboost_train_time": round(xgb_result.train_time, 2),
        "lightgbm_rmse": lgbm_result.rmse,
        "lightgbm_cv_rmse": lgbm_result.cv_rmse,
        "lightgbm_mae": lgbm_result.mae,
        "lightgbm_train_time": round(lgbm_result.train_time, 2),
        "lightgbm_speedup_vs_xgboost": round(speed_ratio, 2),
        "rows_processed": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
    }

    save_results(comparison, results_dir / "model_comparison.json")
    plot_model_comparison(comparison, results_dir / "plots")

    print(f"\n{'='*55}")
    print("  Model Comparison")
    print(f"{'='*55}")
    print(f"  Baseline mean RMSE:  {baseline_result.rmse:.4f}")
    print(f"  Baseline lag24 RMSE: {lag_result.rmse:.4f}")
    print(f"  XGBoost RMSE:        {xgb_result.rmse:.4f}  ({xgb_result.train_time:.1f}s)")
    print(f"  LightGBM RMSE:       {lgbm_result.rmse:.4f}  ({lgbm_result.train_time:.1f}s)")
    if speed_ratio >= 1.0:
        print(f"  LightGBM is {speed_ratio:.1f}x faster than XGBoost")
    else:
        print(f"  XGBoost is {1/speed_ratio:.1f}x faster than LightGBM")
    print(f"{'='*55}\n")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
