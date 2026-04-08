# Metrik AI

Building energy consumption prediction, anomaly detection, and decision support pipeline.

Built for **DS-GA 1019 Advanced Python for Data Science** (Spring 2026, NYU).

## Overview

Metrik AI predicts hourly energy consumption for 1,636 non-residential buildings using the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction) dataset (53.6M rows). The pipeline produces three outputs:

1. **Forecasts** — per-meter hourly consumption predictions using LightGBM
2. **Anomaly scores** — residual-based Modified Z-score detection for abnormal meters
3. **Audit list** — ranked buildings prioritized by anomaly severity and excess consumption

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Build Cython extensions
python setup.py build_ext --inplace

# Download ASHRAE data (requires Kaggle API key)
python scripts/download_data.py
```

### Kaggle API Key

1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create New Token
2. Place `kaggle.json` at `~/.kaggle/kaggle.json`
3. Run `chmod 600 ~/.kaggle/kaggle.json`

## Usage

```bash
# Full pipeline
python -m src.cli run --config config/config.yaml

# Development mode (first 3 chunks only)
python -m src.cli run --n-chunks 3

# Run optimization benchmarks
python -m src.cli benchmark

# Profile the pipeline
python -m src.cli profile

# PySpark pipeline
python -m src.cli spark
```

Or use the Makefile:

```bash
make setup     # Install deps + build Cython
make data      # Download ASHRAE dataset
make run       # Full pipeline
make run-dev   # Quick dev run (3 chunks)
make benchmark # Optimization benchmarks
make test      # Run pytest
```

## Project Structure

```
├── src/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration management
│   ├── load.py             # Chunked data loading
│   ├── features.py         # Feature engineering (vectorized, leakage-safe)
│   ├── model.py            # Baseline + LightGBM training
│   ├── anomaly.py          # Anomaly detection (Modified Z-score)
│   ├── decision.py         # Decision support (ranked audit list)
│   ├── parallel.py         # Multiprocessing + threading
│   ├── numba_ops.py        # Numba JIT-compiled operations
│   ├── cython_kernels.pyx  # Cython-compiled rolling window kernels
│   ├── gpu_ops.py          # CuPy GPU-accelerated operations
│   ├── spark_pipeline.py   # PySpark distributed pipeline
│   └── benchmark.py        # Profiling and benchmarking utilities
├── config/
│   └── config.yaml         # Pipeline configuration
├── scripts/
│   ├── download_data.py    # ASHRAE dataset download
│   └── setup.sh            # One-step setup
├── tests/
│   ├── test_features.py    # Feature leakage tests
│   ├── test_pipeline.py    # Integration tests
│   └── test_numba.py       # Numba correctness tests
├── data/                   # ASHRAE CSV files (not in repo)
├── results/                # Pipeline outputs
├── requirements.txt
├── setup.py                # Cython build
└── Makefile
```

## Configuration

Edit `config/config.yaml` to adjust chunk size, model parameters, anomaly thresholds, and parallelization settings.

## Dataset

ASHRAE Great Energy Predictor III — 53.6M hourly meter readings from 1,636 buildings across 19 sites (2016–2017). See `data/README.md` for download instructions.
