from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession, DataFrame as SparkDF
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def get_spark_session(app_name: str = "MetrikAI", local: bool = True) -> SparkSession:
    if not SPARK_AVAILABLE:
        raise RuntimeError("PySpark not available; install with: pip install pyspark")

    builder = SparkSession.builder.appName(app_name)
    if local:
        builder = builder.master("local[*]")

    builder = (
        builder
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
    )

    return builder.getOrCreate()


def load_train_spark(spark: SparkSession, data_dir: str, row_limit: int | None = None) -> SparkDF:
    t0 = time.perf_counter()

    train = spark.read.csv(
        str(Path(data_dir) / "train.csv"),
        header=True,
        inferSchema=True,
    )
    train = train.withColumn("timestamp", F.to_timestamp("timestamp"))
    if row_limit is not None and row_limit > 0:
        train = train.limit(row_limit)

    meta = spark.read.csv(
        str(Path(data_dir) / "building_metadata.csv"),
        header=True,
        inferSchema=True,
    )

    weather = spark.read.csv(
        str(Path(data_dir) / "weather_train.csv"),
        header=True,
        inferSchema=True,
    )
    weather = weather.withColumn("timestamp", F.to_timestamp("timestamp"))

    df = train.join(meta, on="building_id", how="left")
    df = df.join(weather, on=["site_id", "timestamp"], how="left")

    elapsed = time.perf_counter() - t0
    logger.info("Spark data load: %.2fs, %d rows", elapsed, df.count())
    return df


def add_time_features_spark(df: SparkDF) -> SparkDF:
    return (
        df
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("dayofweek", F.dayofweek("timestamp"))
        .withColumn("month", F.month("timestamp"))
        .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin(1, 7), 1).otherwise(0))
        .withColumn("day_of_year", F.dayofyear("timestamp"))
    )


def add_lag_features_spark(df: SparkDF, lag_hours: list[int] | None = None) -> SparkDF:
    if lag_hours is None:
        lag_hours = [24, 168]

    partition = Window.partitionBy("building_id", "meter").orderBy("timestamp")

    for lag in lag_hours:
        col_name = f"lag_{lag}h"
        df = df.withColumn(col_name, F.lag("meter_reading", lag).over(partition))

    return df


def add_rolling_features_spark(df: SparkDF, windows: list[int] | None = None) -> SparkDF:
    if windows is None:
        windows = [24, 168]

    for w in windows:
        partition = (
            Window
            .partitionBy("building_id", "meter")
            .orderBy("timestamp")
            .rowsBetween(-w, -1)
        )
        df = df.withColumn(f"rolling_mean_{w}h", F.avg("meter_reading").over(partition))
        df = df.withColumn(f"rolling_std_{w}h", F.stddev("meter_reading").over(partition))

    return df


def compute_metrics_spark(df: SparkDF, pred_col: str = "prediction") -> dict:
    from pyspark.ml.evaluation import RegressionEvaluator

    evaluator_rmse = RegressionEvaluator(
        labelCol="meter_reading", predictionCol=pred_col, metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="meter_reading", predictionCol=pred_col, metricName="mae"
    )

    rmse = evaluator_rmse.evaluate(df)
    mae = evaluator_mae.evaluate(df)
    mean_actual = df.agg(F.mean("meter_reading")).first()[0]
    cv_rmse = rmse / mean_actual if mean_actual and mean_actual > 0 else float("inf")

    return {"rmse": rmse, "mae": mae, "cv_rmse": cv_rmse}


def run_spark_pipeline(data_dir: str) -> dict:
    t0 = time.perf_counter()
    spark = get_spark_session()

    try:
        df = load_train_spark(spark, data_dir)
        df = add_time_features_spark(df)
        df = add_lag_features_spark(df)
        df = add_rolling_features_spark(df)

        row_count = df.count()
        elapsed = time.perf_counter() - t0

        logger.info("Spark pipeline: %d rows processed in %.2fs", row_count, elapsed)
        return {"rows": row_count, "elapsed_seconds": elapsed}

    finally:
        spark.stop()
