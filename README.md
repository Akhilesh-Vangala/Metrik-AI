# Metrik AI — Building Energy Consumption Prediction

**DS-GA 1019 · Advanced Python for Data Science · NYU · Spring 2026**  
Group 17 · Akhilesh Vangala (sv3129) · Lucas Yao (ly2808)

End-to-end pipeline for hourly energy forecasting, anomaly detection, and decision support on the ASHRAE Great Energy Predictor III dataset.

---

## Results Summary

| Metric | Value |
|---|---|
| Raw training rows | 20,216,100 |
| Clean rows after feature engineering | 7,191,724 |
| Features engineered | 38 |
| Baseline (per-meter mean) RMSE | 497.4 |
| Baseline (lag-24h) RMSE | 238.8 |
| XGBoost RMSE | 184.9 |
| **LightGBM RMSE** | **183.9** |
| RMSE improvement over baseline | **63%** |
| Anomalies flagged | 68,504 (3.4% of val set) |
| Buildings in audit list | 964 |
| Full pipeline runtime | ~96 seconds |
| Parallel speedup (8 workers, 16 sites) | 1.62× |

---

## Reproducing the Project

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Build Cython extension

```bash
python setup.py build_ext --inplace
```

This compiles `src/cython_kernels.pyx` → `src/cython_kernels.so`. Required for the benchmark command.

### Step 3 — Download the dataset

The pipeline uses the **ASHRAE Great Energy Predictor III** dataset from Kaggle.

The pipeline only needs three files: `train.csv`, `building_metadata.csv`, and `weather_train.csv`.

**Option A — Kaggle CLI (recommended):**

```bash
pip install kaggle
```

1. Go to kaggle.com → Account → **Create New API Token** → download `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`
3. Accept the competition rules at kaggle.com/competitions/ashrae-energy-prediction/rules
4. Run:

```bash
python3 scripts/download_data.py
```

**Option B — Manual download:**

Go to kaggle.com/competitions/ashrae-energy-prediction/data, log in, and download these three files into the `data/` directory:

```
data/
├── train.csv               (647 MB — 20.2M rows, required)
├── building_metadata.csv   (44 KB, required)
└── weather_train.csv       (14 MB, required)
```

The remaining files (`test.csv`, `weather_test.csv`, `sample_submission.csv`) are not used by the pipeline and do not need to be downloaded.

### Step 4 — Run the full pipeline

```bash
python -m src.cli run
```

This runs all 8 stages in sequence: chunked loading → cleaning → feature engineering → train/val split → baseline models → LightGBM → anomaly detection → audit list → plots.

Runtime: ~96 seconds on a standard laptop.

### Step 5 — Run EDA

```bash
python -m src.cli eda
```

Generates 7 exploratory plots into `results/eda/`.

### Step 6 — Run optimization benchmarks

```bash
python -m src.cli benchmark
```

Benchmarks Python loop vs NumPy vs Numba vs Cython at three data sizes. Generates `results/benchmarks.csv` and `results/plots/benchmark_speedups.png`.

### Step 7 — Run LightGBM vs XGBoost comparison

```bash
python -m src.cli compare
```

Trains both LightGBM and XGBoost on the full dataset and compares RMSE and training time. Generates `results/model_comparison.json`.

### Step 8 — Run parallel training benchmark

```bash
python -m src.cli parallel-benchmark
```

Trains one LightGBM model per site (16 sites) sequentially then in parallel with 2/4/8/16 workers. Generates `results/parallel_benchmark.csv` and the speedup plot.

### Step 9 — Run profiling

```bash
python -m src.cli profile
```

Runs cProfile on the pipeline and reports top-30 cumulative time functions.

### Step 10 — Run tests

```bash
python -m pytest tests/ -v
```

---

## All-at-once (Makefile)

```bash
make setup          # install deps + build Cython
make run            # full pipeline
make eda            # EDA plots
make benchmark      # optimization benchmarks
make compare        # LightGBM vs XGBoost
make parallel-benchmark  # parallel speedup curve
make profile        # cProfile report
make test           # run test suite
```

Or run everything sequentially:

```bash
make all
```

---

## CLI Reference

```
python -m src.cli [OPTIONS] COMMAND

Options:
  --config PATH   Path to config YAML (default: config/config.yaml)
  -v, --verbose   Enable debug logging

Commands:
  run                  Full pipeline (load → clean → features → train → anomaly → audit)
  eda                  Exploratory data analysis and plots
  benchmark            Optimization benchmarks (Python / NumPy / Numba / Cython / GPU)
  compare              LightGBM vs XGBoost accuracy and speed comparison
  parallel-benchmark   Parallel training speedup across 1/2/4/8/16 workers
  profile              cProfile + tracemalloc memory profiling
  quality              Per-site data quality report
  spark                PySpark distributed pipeline (requires Spark installation)
```

All commands support `--n-chunks N` to limit data for faster development runs:

```bash
python -m src.cli run --n-chunks 6       # ~6M rows, fast dev run (needs ≥6 chunks for valid train/val split)
python -m src.cli benchmark              # always uses synthetic data, always fast
```

---

## Outputs

All results are saved to `results/`:

| File | Description |
|---|---|
| `pipeline_results.json` | Full metrics: RMSE, MAE, CV-RMSE, feature importance, timing |
| `model_comparison.json` | LightGBM vs XGBoost: RMSE, CV-RMSE, MAE, training time |
| `model.lgb` | Trained LightGBM model in native format (loadable via `lgb.Booster`) |
| `predictions.csv` | Validation predictions: building_id, meter, timestamp, actual, predicted, residual |
| `anomaly_summary.csv` | Per-meter anomaly rate, max streak length, excess consumption |
| `audit_list.csv` | 964 buildings ranked by composite priority score |
| `data_quality.csv` | Per-site: row counts, missingness %, zero reading % |
| `benchmarks.csv` | Speedup table: Python loop / NumPy / Numba / Cython at 100K–1M rows |
| `parallel_benchmark.csv` | Workers (1/2/4/8/16) → time → speedup across 16 sites |
| `eda/` | 7 EDA plots (distributions, temporal patterns, weather correlations) |
| `plots/` | 6 pipeline plots (feature importance, predictions vs actual, anomaly distribution, model comparison, benchmark speedups, parallel speedup) |
| `profiling/pipeline.prof` | Binary cProfile output (open with `snakeviz results/profiling/pipeline.prof`) |

---

## Dataset

**ASHRAE Great Energy Predictor III** (Miller et al., 2020, *Scientific Data*)

| Attribute | Value |
|---|---|
| Raw training rows | 20,216,100 hourly records |
| Clean rows (after outlier removal + lag NaN drop) | ~7.2M |
| Buildings | 1,449 |
| Meters | 2,380 (electricity, chilled water, steam, hot water) |
| Sites | 16 across North America and Europe |
| Period | January 2016 – December 2016 |

**Why raw 20.2M becomes ~7.2M clean rows:**  
Each of the 2,380 meters loses its first 168 hours (7 days) of data due to the 168-hour rolling window features. Combined with outlier removal and weather NaN imputation gaps, ~13M rows are dropped. The pipeline reads all 20.2M raw rows — the reduction is a consequence of feature engineering, not data truncation.

**Known issues handled at load time:**
- Site 0 electricity readings are in kBTU, not kWh — converted automatically
- Zero-reading streaks ≥ 48 hours indicate meter outages — flagged and removed
- Extreme outliers capped at 99.9th percentile per building/meter
- Weather columns have 10–40% missing values — imputed per-site via forward fill

---

## Pipeline Architecture

```
train.csv (20.2M rows)
    │
    ▼ Chunked I/O — 2M rows/chunk, ThreadPoolExecutor for I/O, itertools.islice for limits
    │
    ▼ Data Cleaning — kBTU conversion, outlier capping, zero-streak detection
    │
    ▼ Feature Engineering — 38 features: time cyclicals, lag-24h/168h, rolling stats,
    │                        weather interactions, building metadata (vectorized NumPy/pandas)
    │
    ▼ Time-Based Split — last 3 months of 2016 → validation (no data leakage)
    │
    ├── Baseline Mean (per-meter historical mean)  → RMSE 497.4
    ├── Baseline Lag-24h                            → RMSE 238.8
    ├── XGBoost (hist, 1000 rounds max, early stopping) → RMSE 184.9
    └── LightGBM (63 leaves, early stopping)        → RMSE 183.9  ← deployed
              │
              ▼ Residuals → Modified Z-score (MAD) per meter
              │
              ▼ Anomaly flags → temporal clustering → severity scoring
              │
              ▼ Composite priority score → ranked audit list (964 buildings)
```

---

## Advanced Python Techniques

Every topic in the DS-GA 1019 curriculum is implemented as a working pipeline component:

| Course Week | Topic | Implementation | File |
|---|---|---|---|
| 2 | Python Performance Tips | `__slots__` on all config dataclasses; float32 dtypes; pre-allocated arrays | `config.py`, `features.py` |
| 3 | itertools | `product` for (site, meter) dispatch; `islice` for chunk limiting; `accumulate`, `chain` | `parallel.py`, `load.py` |
| 4 | Performance Tuning | Vectorized feature build vs row-by-row baseline — benchmarked at 5K/10K/50K rows | `features.py`, `benchmark.py` |
| 5 | Cython | Custom kernels compiled to C with typed memoryviews and `boundscheck=False` | `cython_kernels.pyx` |
| 6 | Numba | `@njit(parallel=True)` for anomaly scoring (275M rows/sec rolling mean) | `numba_ops.py` |
| 8 | Optimization in Python | `scipy.optimize.minimize_scalar` for LightGBM learning rate search | `model.py` |
| 9 | Python Concurrency | `ThreadPoolExecutor` (I/O), `ProcessPoolExecutor` (training), `Queue`/`Lock`/`Thread` | `parallel.py` |
| 10–11 | Parallel Programming | Per-site training across 16 sites; 1.62× speedup at 8 workers | `parallel.py`, `cli.py` |
| 12 | Python for GPUs | CuPy anomaly scoring + array ops; graceful CPU fallback when GPU unavailable | `gpu_ops.py` |
| 13 | BigData with PySpark | Distributed pipeline with Spark SQL window functions for lag/rolling features | `spark_pipeline.py` |

**Note on GPU and Spark:** Both are fully implemented with correct logic. On machines without a CUDA GPU or Spark installation, they gracefully fall back with a clear error message. The code paths are correct and testable on appropriate hardware.

---

## Benchmark Results

### Anomaly Scoring (1M rows)

| Method | Throughput | Speedup vs Python |
|---|---|---|
| Python loop | 1.5M rows/sec | 1× |
| NumPy vectorized | 55M rows/sec | **37×** |
| Numba JIT | 7.8M rows/sec | 5× |
| Cython | 23M rows/sec | 15× |

### Rolling Mean (500K rows)

| Method | Throughput |
|---|---|
| pandas | 127M rows/sec |
| Numba JIT | **440M rows/sec** |

### Parallel Training (16 sites)

| Workers | Time | Speedup |
|---|---|---|
| 1 (sequential) | 76.6s | 1.0× |
| 2 | 60.4s | 1.27× |
| 4 | 53.5s | 1.43× |
| 8 | 47.4s | **1.62×** |
| 16 | 47.8s | 1.60× |

Speedup plateaus at 8–16 workers because each per-site LightGBM already uses multiple threads internally.

---

## Project Structure

```
metrik-ai/
├── src/
│   ├── cli.py               CLI entry point — 8 subcommands via Click
│   ├── config.py            YAML config → typed dataclasses (slots=True)
│   ├── load.py              Chunked I/O, data cleaning, quality reporting
│   ├── features.py          Vectorized + naive feature engineering (38 features)
│   ├── model.py             Baselines, LightGBM, XGBoost, per-site training
│   ├── anomaly.py           Modified Z-score + temporal streak clustering
│   ├── decision.py          Composite priority scoring → audit list
│   ├── eda.py               EDA plots and summary statistics
│   ├── parallel.py          ProcessPool + ThreadPool + itertools dispatch
│   ├── numba_ops.py         @njit(parallel=True) kernels
│   ├── cython_kernels.pyx   Cython C-extension (build with setup.py)
│   ├── gpu_ops.py           CuPy GPU operations with CPU fallback
│   ├── spark_pipeline.py    PySpark distributed pipeline
│   └── benchmark.py         BenchmarkSuite, cProfile wrapper, tracemalloc
├── tests/
│   ├── test_features.py     Leakage safety, time features, dtype enforcement
│   ├── test_pipeline.py     Anomaly detection, time split, end-to-end mini pipeline
│   └── test_numba.py        Numba correctness vs NumPy reference
├── config/
│   └── config.yaml          All pipeline parameters
├── scripts/
│   ├── download_data.py     Kaggle API dataset download
│   └── setup.sh             One-command environment setup
├── data/
│   └── README.md            Download instructions (CSVs are gitignored)
├── results/                 All pipeline outputs (pre-run results committed)
├── requirements.txt
├── setup.py                 Cython build configuration
└── Makefile
```

---

## Configuration

```yaml
# config/config.yaml
pipeline:
  chunk_size: 2_000_000      # rows per chunk (out-of-core loading)
  seed: 42
  validation_months: 3       # last N months held out for validation

model:
  lgbm_params:
    num_leaves: 63
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
  num_boost_round: 1000
  early_stopping_rounds: 50

anomaly:
  threshold: 3.5             # modified Z-score cutoff
  min_hours: 100             # minimum readings for reliable scoring

parallel:
  n_workers: 4
```

---

## Tests

```bash
python -m pytest tests/ -v
```

21 tests covering leakage safety, Numba correctness against NumPy, and an end-to-end mini pipeline on synthetic data.

---

## References

1. Miller et al. (2020). *The Building Data Genome Project 2.* Scientific Data 7, 368.
2. Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
3. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
4. Runge & Zmeureanu (2019). *Forecasting Energy Use in Buildings.* Energies 12(18), 3355.
5. Molina-Solana et al. (2017). *Data Science for Building Energy Management.* Renewable and Sustainable Energy Reviews 70, 598–609.
