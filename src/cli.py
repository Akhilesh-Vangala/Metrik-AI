from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.config import load_config
from src.load import load_building_metadata, load_weather, stream_train_chunks, load_full_dataset
from src.features import build_features, get_feature_columns
from src.model import time_based_split, train_baseline_mean, train_baseline_lag, train_lightgbm, compute_residuals
from src.anomaly import detect_anomalies, aggregate_anomalies
from src.decision import build_audit_list, export_audit_list
from src.benchmark import BenchmarkSuite, profile_function, measure_memory


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
@click.option("--n-chunks", type=int, default=None, help="Limit number of chunks (for dev)")
@click.pass_context
def run(ctx, n_chunks: int | None):
    """Run the full pipeline: load -> features -> model -> anomaly -> audit."""
    cfg = ctx.obj["cfg"]
    t_start = time.perf_counter()

    logging.getLogger(__name__).info("Starting Metrik AI pipeline")

    meta = load_building_metadata(cfg)
    weather = load_weather(cfg)

    frames = []
    for chunk in stream_train_chunks(cfg, meta, weather, n_chunks=n_chunks):
        chunk = build_features(chunk, cfg.features)
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    del frames

    feat_cols = get_feature_columns(df)
    df = df.dropna(subset=feat_cols + [cfg.pipeline.target_col])

    train_df, val_df = time_based_split(df, cfg.pipeline.validation_months)

    baseline_result = train_baseline_mean(train_df, val_df, cfg.pipeline.target_col)
    lag_result = train_baseline_lag(train_df, val_df, cfg.pipeline.target_col)
    lgbm_result = train_lightgbm(train_df, val_df, cfg)

    residuals = compute_residuals(val_df[cfg.pipeline.target_col].values, lgbm_result.predictions)
    val_df = val_df.copy()
    val_df["residual"] = residuals

    val_df = detect_anomalies(val_df, cfg=cfg.anomaly)
    anomaly_summary = aggregate_anomalies(val_df)

    audit = build_audit_list(anomaly_summary, meta, min_hours=cfg.anomaly.min_hours)
    output_path = cfg.paths.results_path() / "audit_list.csv"
    export_audit_list(audit, output_path)

    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"Metrik AI Pipeline Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Rows processed:   {len(df):,}")
    print(f"  Baseline RMSE:    {baseline_result.rmse:.4f}")
    print(f"  Lag24 RMSE:       {lag_result.rmse:.4f}")
    print(f"  LightGBM RMSE:    {lgbm_result.rmse:.4f}")
    print(f"  RMSE improvement: {(1 - lgbm_result.rmse / baseline_result.rmse) * 100:.1f}%")
    print(f"  Anomalies found:  {val_df['is_anomaly'].sum():,}")
    print(f"  Audit list:       {output_path}")
    print(f"{'='*60}\n")


@cli.command()
@click.pass_context
def benchmark(ctx):
    """Run optimization benchmarks and generate comparison tables."""
    from src.numba_ops import modified_zscore_numba, warmup as numba_warmup
    from src.anomaly import modified_zscore_naive, modified_zscore_vectorized

    cfg = ctx.obj["cfg"]
    suite = BenchmarkSuite()

    for size in [100_000, 500_000, 1_000_000]:
        data = np.random.randn(size).astype(np.float64)

        suite.time_function("anomaly_scoring", "python_loop", modified_zscore_naive, data, input_size=size)
        suite.time_function("anomaly_scoring", "numpy_vectorized", modified_zscore_vectorized, data, input_size=size)

        numba_warmup()
        suite.time_function("anomaly_scoring", "numba_jit", modified_zscore_numba, data, input_size=size)

    suite.print_summary()

    results_path = cfg.paths.results_path() / "benchmarks.csv"
    pd.DataFrame(suite.summary_table()).to_csv(results_path, index=False)
    print(f"\nBenchmarks saved to {results_path}")


@cli.command()
@click.pass_context
def profile(ctx):
    """Profile the pipeline and save results."""
    cfg = ctx.obj["cfg"]
    output_dir = Path(cfg.profiling.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def pipeline_subset():
        meta = load_building_metadata(cfg)
        weather = load_weather(cfg)
        frames = list(stream_train_chunks(cfg, meta, weather, n_chunks=2))
        df = pd.concat(frames, ignore_index=True)
        build_features(df, cfg.features)

    report = profile_function(pipeline_subset, output_path=str(output_dir / "pipeline.prof"))
    print(report)


@cli.command()
@click.pass_context
def spark(ctx):
    """Run PySpark pipeline on the full dataset."""
    from src.spark_pipeline import run_spark_pipeline
    cfg = ctx.obj["cfg"]
    result = run_spark_pipeline(cfg.paths.data_dir)
    print(f"Spark pipeline: {result['rows']:,} rows in {result['elapsed_seconds']:.1f}s")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
