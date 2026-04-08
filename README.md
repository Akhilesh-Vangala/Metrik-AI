# Metrik AI

Building energy consumption prediction, anomaly detection, and decision support pipeline.

Built for **DS-GA 1019 Advanced Python for Data Science** (Spring 2026, NYU).

## Overview

Metrik AI predicts hourly energy consumption for 1,636 non-residential buildings using the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction) dataset (53.6M rows). The pipeline produces three outputs:

1. **Forecasts** — per-meter hourly consumption predictions using LightGBM
2. **Anomaly scores** — per-meter residual-based Modified Z-score with temporal clustering
3. **Audit list** — ranked buildings prioritized by anomaly severity, streak length, and excess consumption

## Setup

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
python scripts/download_data.py
```

### Kaggle API Key

1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create New Token
2. Place `kaggle.json` at `~/.kaggle/kaggle.json`
3. Run `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### CLI Commands

```bash
python -m src.cli run                    # Full pipeline (load → clean → features → model → anomaly → audit)
python -m src.cli run --n-chunks 3       # Dev mode (first 3 chunks only)
python -m src.cli benchmark              # Optimization benchmarks (anomaly, features, rolling, memory)
python -m src.cli parallel-benchmark     # Parallel speedup curve (1/2/4/8 workers)
python -m src.cli profile               # cProfile + memory profiling
python -m src.cli spark                  # PySpark distributed pipeline
python -m src.cli quality               # Data quality report
```

### Makefile

```bash
make setup               # Install deps + build Cython
make data                # Download ASHRAE dataset
make test                # Run pytest (19 tests)
make run                 # Full pipeline
make run-dev             # Quick dev run (3 chunks)
make benchmark           # Optimization benchmarks
make parallel-benchmark  # Parallel speedup curve
make profile             # Pipeline profiling
make spark               # PySpark pipeline
make quality             # Data quality report
```

## Pipeline Stages

| Stage | Description | Key Technique |
|-------|-------------|---------------|
| **1. Data Loading** | Chunked CSV reading (2M rows/chunk), merge metadata + weather | Out-of-core I/O, `itertools.islice` |
| **2. Data Cleaning** | Site 0 kBTU→kWh, outlier capping, zero-streak detection | Vectorized pandas, domain knowledge |
| **3. Feature Engineering** | Time, lag, rolling, weather, interaction features (30+ total) | Vectorized groupby+shift+rolling |
| **4. Train/Val Split** | Time-based split (last 3 months = validation) | Leakage prevention |
| **5. Model Training** | Baseline mean, baseline lag-24h, LightGBM | Early stopping, categorical features |
| **6. Anomaly Detection** | Per-meter Modified Z-score, temporal streak clustering | Numba JIT acceleration |
| **7. Decision Support** | Weighted priority scoring, ranked audit list | Composite score: rate + severity + excess |

## Outputs

All results saved to `results/`:

| File | Description |
|------|-------------|
| `pipeline_results.json` | Metrics, feature importance, timing |
| `predictions.csv` | Validation predictions with residuals |
| `anomaly_summary.csv` | Per-meter anomaly statistics |
| `audit_list.csv` | Ranked buildings to investigate |
| `data_quality.csv` | Per-site data quality report |
| `model.lgb` | Trained LightGBM model |
| `benchmarks.csv` | Optimization benchmark results |
| `parallel_benchmark.csv` | Parallel speedup measurements |

## Advanced Python Techniques

| Course Topic | Implementation |
|---|---|
| Performance Tips (Week 2) | `__slots__` dataclasses, float32 dtype, pre-allocated arrays |
| itertools (Week 3) | `itertools.product` for work dispatch, `itertools.islice` for chunking |
| Performance Tuning (Week 4) | Vectorized feature engineering vs. naive iterrows baseline |
| Cython (Week 5) | `cython_kernels.pyx`: rolling mean/std, modified z-score |
| Numba (Week 6) | `numba_ops.py`: @njit parallel anomaly scoring, rolling windows |
| Optimization (Week 8) | `lru_cache` for holidays, dict-based lookups, algorithmic choices |
| Concurrency (Week 9) | ThreadPoolExecutor (I/O), ProcessPoolExecutor (CPU) |
| Parallel Programming (Weeks 10-11) | Per-site model training with speedup curve |
| GPUs (Week 12) | CuPy-based anomaly scoring and array ops |
| PySpark (Week 13) | Distributed pipeline with Spark SQL window functions |

## Project Structure

```
├── src/
│   ├── cli.py              # Click CLI (6 subcommands)
│   ├── config.py           # YAML config with typed dataclasses
│   ├── load.py             # Chunked loading + data cleaning
│   ├── features.py         # Vectorized + naive feature engineering
│   ├── model.py            # Baselines + LightGBM + per-site training
│   ├── anomaly.py          # Per-meter z-score + temporal clustering
│   ├── decision.py         # Priority scoring + audit list
│   ├── parallel.py         # Process/thread pool dispatch
│   ├── numba_ops.py        # Numba JIT kernels
│   ├── cython_kernels.pyx  # Cython compiled kernels
│   ├── gpu_ops.py          # CuPy GPU operations
│   ├── spark_pipeline.py   # PySpark distributed pipeline
│   └── benchmark.py        # Profiling + benchmarking utilities
├── config/config.yaml
├── tests/ (19 tests)
├── scripts/
├── data/ (not in repo)
├── results/
├── requirements.txt
├── setup.py
└── Makefile
```

## Configuration

Edit `config/config.yaml` to adjust chunk size, model hyperparameters, anomaly thresholds, parallelization workers, and profiling settings.

## Dataset

[ASHRAE Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction) — 53.6M hourly meter readings from 1,636 buildings across 19 sites (2016–2017). See `data/README.md` for download instructions.
