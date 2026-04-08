# Metrik AI

**Predicting and Optimizing Building Energy Consumption at Scale**

A production-grade Python pipeline for hourly energy forecasting, anomaly detection, and decision support — built on the ASHRAE Great Energy Predictor III dataset (20.2M training rows, 1,449 buildings, 16 sites).

Developed for **DS-GA 1019 · Advanced Python for Data Science · NYU · Spring 2026**

---

## Problem

Commercial buildings consume ~40% of all U.S. energy and waste 20–30% of it. Operators rely on fixed schedules instead of data-driven control. The cost of building energy waste exceeds $130 billion globally each year.

**Metrik AI** addresses this by predicting hourly energy consumption at the individual meter level, flagging anomalous behavior, and surfacing a prioritized audit list — all processed at scale without exceeding commodity memory limits.

## Solution Architecture

The pipeline decomposes the problem into three components that mirror the real-world Measurement & Verification (M&V) workflow used by energy services companies:

```
train.csv (20.2M rows)
    │
    ▼ [1] Chunked I/O (2M rows/chunk, ~4 GB peak)
    │
    ▼ [2] Data Cleaning (Site 0 kBTU→kWh, outlier capping, zero-streak detection)
    │
    ▼ [3] Feature Engineering (30+ features: time, lag, rolling, weather, interactions)
    │
    ▼ [4] Time-Based Split (last 3 months → validation)
    │
    ├──▶ [5a] Baseline Models (per-meter mean, lag-24h)
    └──▶ [5b] LightGBM Forecasting (early stopping, categorical features)
              │
              ▼ [6] Anomaly Detection (per-meter Modified Z-score + temporal clustering)
              │
              ▼ [7] Decision Support (weighted priority scoring → ranked audit list)
              │
              ▼ [8] Visualizations (feature importance, residuals, anomaly distributions)
```

| Component | Method | Output |
|-----------|--------|--------|
| **Forecasting** | LightGBM with leakage-safe features | Per-meter hourly predictions |
| **Anomaly Detection** | Per-meter Modified Z-score with streak clustering | Anomaly scores + binary flags |
| **Decision Support** | Weighted composite scoring (rate + severity + excess + streaks) | Ranked audit list |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Cython Extensions (optional, for benchmarking)

```bash
python setup.py build_ext --inplace
```

### 3. Download Dataset

**Option A — Kaggle CLI (recommended):**

```bash
pip install kaggle
```

1. Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New Token**
2. Place `kaggle.json` at `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`
4. Accept competition rules at [kaggle.com/competitions/ashrae-energy-prediction/rules](https://www.kaggle.com/competitions/ashrae-energy-prediction/rules)
5. Run:

```bash
python scripts/download_data.py
```

**Option B — Manual:**

Download from [kaggle.com/competitions/ashrae-energy-prediction/data](https://www.kaggle.com/competitions/ashrae-energy-prediction/data) and unzip into `data/`.

### 4. Run the Pipeline

```bash
python -m src.cli run                    # Full pipeline
python -m src.cli run --n-chunks 3       # Dev mode (first 3 chunks)
```

---

## CLI Reference

All commands are available through `python -m src.cli` or `make`:

| Command | Description | Make Target |
|---------|-------------|-------------|
| `run` | Full pipeline: load → clean → features → train → anomaly → audit | `make run` |
| `run --n-chunks N` | Dev mode with limited chunks | `make run-dev` |
| `eda` | Exploratory data analysis with plots | `make eda` |
| `benchmark` | Comprehensive optimization benchmarks | `make benchmark` |
| `parallel-benchmark` | Parallel speedup curve (1/2/4/8 workers) | `make parallel-benchmark` |
| `profile` | cProfile + memory profiling | `make profile` |
| `quality` | Data quality report per site | `make quality` |
| `spark` | PySpark distributed pipeline | `make spark` |

All commands accept `--config path/to/config.yaml` and `-v` (verbose) flags.

---

## Pipeline Outputs

All results are saved to `results/`:

| File | Description |
|------|-------------|
| `pipeline_results.json` | Full metrics: RMSE, MAE, CV-RMSE, feature importance, timing |
| `predictions.csv` | Validation set: building_id, meter, timestamp, actual, predicted, residual |
| `anomaly_summary.csv` | Per-meter: anomaly_rate, max_streak, excess_consumption |
| `audit_list.csv` | Ranked buildings by composite priority score |
| `data_quality.csv` | Per-site: row counts, missingness, zero percentages |
| `model.lgb` | Trained LightGBM model (native format, loadable) |
| `benchmarks.csv` | Optimization benchmark results with speedups |
| `parallel_benchmark.csv` | Worker count → time → speedup measurements |
| `eda/` | EDA plots: distributions, temporal patterns, correlations |
| `plots/` | Pipeline plots: feature importance, predictions vs actual, anomaly distribution |

---

## Dataset

**ASHRAE Great Energy Predictor III** — the largest publicly available benchmark for building energy prediction. Peer-reviewed in Miller et al. (2020), *Scientific Data* 7, 368.

| Attribute | Value |
|-----------|-------|
| Training rows | 20,216,100 hourly records |
| Buildings | 1,449 non-residential |
| Meters | 2,380 (electricity, chilled water, steam, hot water) |
| Sites | 16 (North America and Europe) |
| Training period | January 2016 – December 2016 (1 year) |
| Test period | January 2017 – May 2018 (41.7M rows, no labels) |
| Total size | ~2.6 GB (all files) |

**Known data issues handled by our pipeline:**
- Site 0 electricity is in kBTU (not kWh) — converted at load time
- Zero-reading streaks (≥48h) indicate meter outages — flagged as NaN
- Extreme outliers capped at 99.9th percentile per building/meter
- Weather columns have 10–40% missing values — imputed per-site

---

## Advanced Python Techniques

Every major topic from DS-GA 1019 is applied as a load-bearing component:

| Week | Topic | Implementation | File |
|------|-------|----------------|------|
| 2 | Performance Tips | `__slots__` on all 8 config dataclasses; float32 dtypes throughout; pre-allocated arrays | `config.py` |
| 3 | itertools | `itertools.product` for (site, meter) work dispatch; `itertools.islice` for chunk limiting | `parallel.py`, `load.py` |
| 4 | Performance Tuning | Vectorized feature engineering (30+ features) vs naive row-by-row baseline | `features.py` |
| 5 | Cython | Rolling mean, rolling std, modified z-score compiled to C with typed memoryviews | `cython_kernels.pyx` |
| 6 | Numba | `@njit(parallel=True)` anomaly scoring, rolling windows, flag computation | `numba_ops.py` |
| 8 | Optimization | `lru_cache` for holiday lookups; dict-based weather joins; algorithmic choices | `features.py`, `load.py` |
| 9 | Concurrency | `ThreadPoolExecutor` for I/O-bound chunk reading; `ProcessPoolExecutor` for CPU-bound training | `parallel.py` |
| 10–11 | Parallel Programming | Per-site model training with measured speedup curves (1/2/4/8 workers) | `parallel.py`, `model.py` |
| 12 | GPUs | CuPy-based anomaly scoring and array operations (NumPy-compatible API) | `gpu_ops.py` |
| 13 | PySpark | Distributed pipeline with Spark SQL window functions for lag/rolling features | `spark_pipeline.py` |

**Profiling & Benchmarking:**
- `cProfile` for CPU profiling with top-30 cumulative time report
- `tracemalloc` for peak memory measurement
- `BenchmarkSuite` class with automatic speedup computation
- Comprehensive benchmark command testing all techniques at multiple data sizes

---

## Benchmark Targets

| Component | Baseline | Optimized | Target |
|-----------|----------|-----------|--------|
| Data load | Full in-memory read (OOM) | Chunked 2M rows/chunk | No OOM, <4 GB peak |
| Feature build (5M rows) | Row-by-row `iterrows()` | Vectorized pandas/NumPy | ≥10× speedup |
| Forecasting | Per-meter mean | LightGBM | ≥20% RMSE improvement |
| Anomaly scoring (1M) | Pure Python loop | Numba JIT | ≥5× speedup |
| Parallel training | Sequential (1 worker) | ProcessPool (4 workers) | Near-linear speedup |
| Memory | float64 arrays | float32 arrays | ~50% reduction |

---

## Project Structure

```
metrik-ai/
├── src/                        # Core pipeline package
│   ├── __init__.py             # Package marker (version)
│   ├── __main__.py             # python -m src entry point
│   ├── cli.py                  # Click CLI with 7 subcommands
│   ├── config.py               # YAML config → typed dataclasses (slots=True)
│   ├── load.py                 # Chunked I/O, data cleaning, quality reporting
│   ├── features.py             # Vectorized + naive feature engineering
│   ├── model.py                # Baselines + LightGBM + per-site training + save/load
│   ├── anomaly.py              # Per-meter z-score + temporal streak clustering
│   ├── decision.py             # Composite priority scoring + audit list export
│   ├── eda.py                  # Exploratory analysis + visualization generation
│   ├── parallel.py             # ProcessPool + ThreadPool dispatch with itertools
│   ├── numba_ops.py            # @njit(parallel=True) kernels: z-score, rolling, flags
│   ├── cython_kernels.pyx      # Cython: rolling mean/std, z-score (boundscheck=False)
│   ├── gpu_ops.py              # CuPy GPU operations (graceful fallback if unavailable)
│   ├── spark_pipeline.py       # PySpark distributed pipeline with Window functions
│   └── benchmark.py            # BenchmarkSuite, cProfile wrapper, tracemalloc
├── config/
│   └── config.yaml             # All pipeline parameters (paths, model, anomaly, parallel)
├── tests/
│   ├── test_features.py        # Leakage safety, time features, dtype enforcement
│   ├── test_pipeline.py        # Anomaly detection, time split, end-to-end mini pipeline
│   └── test_numba.py           # Numba correctness vs NumPy reference implementation
├── scripts/
│   ├── download_data.py        # Kaggle CLI dataset download + validation
│   └── setup.sh                # One-command environment setup
├── data/                       # ASHRAE CSV files (gitignored, ~2-3 GB)
│   └── README.md               # Download instructions
├── results/                    # Pipeline outputs (gitignored except .gitkeep)
│   └── .gitkeep
├── requirements.txt            # Pinned dependencies
├── setup.py                    # Cython build configuration
├── Makefile                    # All commands as make targets
├── .gitignore
└── README.md
```

---

## Configuration

All parameters are in `config/config.yaml`:

```yaml
pipeline:
  chunk_size: 2_000_000        # Rows per chunk for out-of-core loading
  seed: 42                     # Random seed for reproducibility
  validation_months: 3         # Last N months used as validation set

model:
  lgbm_params:
    num_leaves: 63
    learning_rate: 0.05
  num_boost_round: 1000
  early_stopping_rounds: 50

anomaly:
  threshold: 3.5               # Modified Z-score threshold for flagging
  min_hours: 100               # Minimum data points for reliable scoring

parallel:
  n_workers: 4                 # ProcessPool worker count
```

---

## Testing

```bash
python -m pytest tests/ -v
```

19 tests covering:
- **Leakage safety**: Lag and rolling features contain no future information
- **Correctness**: Numba JIT outputs match NumPy reference within tolerance
- **Integration**: End-to-end mini pipeline on synthetic data
- **Edge cases**: Constant arrays, single elements, empty groups

---

## References

1. Miller, C., et al. (2020). *The Building Data Genome Project 2.* Scientific Data 7, 368.
2. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
3. Runge, J., & Zmeureanu, R. (2019). *Forecasting Energy Use in Buildings.* Energies 12(18), 3355.
4. Molina-Solana, M., et al. (2017). *Data Science for Building Energy Management.* Renewable and Sustainable Energy Reviews 70, 598–609.

---

## License

Academic project — DS-GA 1019, NYU Center for Data Science.
