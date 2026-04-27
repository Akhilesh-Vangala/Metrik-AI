# Metrik AI

DS-GA 1019 — Advanced Python for Data Science, NYU Spring 2026  
Akhilesh Vangala (sv3129) · Lucas Yao (ly2808)

Building energy consumption forecasting on the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction) dataset. We trained a LightGBM model to forecast hourly meter readings across 1,449 buildings and built an anomaly detection system to flag unusual consumption patterns.

## Results

- LightGBM RMSE: **183.9**
  - 63% lower than the meter-mean baseline (497.4) — exceeds the proposal's 20–40% target
  - 23% lower than the lag-24h persistence baseline (238.8)
- XGBoost RMSE: 184.9
- 68,504 anomalies flagged in the validation set
- 964 buildings in the audit list
- Runs in ~96 seconds end to end

## Setup

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

The Cython build step compiles `src/cython_kernels.pyx` — needed for the benchmark command.

You'll need three files from the [ASHRAE competition](https://www.kaggle.com/competitions/ashrae-energy-prediction/data): `train.csv`, `building_metadata.csv`, and `weather_train.csv`, placed under `data/`. Kaggle CLI works:

```bash
pip install kaggle
# put your kaggle.json at ~/.kaggle/kaggle.json first
python3 scripts/download_data.py
```

Or just download the three files manually from the competition page.

## Running

```bash
python -m src.cli run                  # full pipeline
python -m src.cli eda                  # EDA plots
python -m src.cli benchmark            # Python / NumPy / Numba / Cython speedups
python -m src.cli compare              # LightGBM vs XGBoost
python -m src.cli parallel-benchmark   # parallel training speedup
python -m src.cli profile              # cProfile report
pytest tests/ -v                       # run tests
```

For a faster dev run: `python -m src.cli run --n-chunks 6`

We committed all pre-run results to `results/` so you can see the outputs without re-running.

## What's in results/

| File | What it is |
|---|---|
| `pipeline_results.json` | main metrics — RMSE, MAE, feature importance, timing |
| `model_comparison.json` | LightGBM vs XGBoost side by side |
| `audit_list.csv` | 964 buildings ranked by anomaly severity |
| `anomaly_summary.csv` | per-meter anomaly rates and streak lengths |
| `benchmarks.csv` | speedup table across implementations |
| `parallel_benchmark.csv` | parallel training time by worker count |
| `plots/` | all figures |
| `model.lgb` | trained LightGBM model |

## About the data

ASHRAE Great Energy Predictor III — 20.2M hourly records, 1,449 buildings, 16 sites, all of 2016.

The raw 20.2M rows drop to ~7.2M after outlier removal and lag NaN drops (the 168-hour rolling window needs a full week of history per meter before it can produce a valid row). A few things we had to handle:

- Site 0 electricity is in kBTU not kWh — converted at load time
- Zero-reading streaks ≥ 48h are meter outages, not real zeros — removed
- Outliers capped at 99.9th percentile per building/meter
- Weather columns have 10–40% missingness — forward filled per site

## Implementation

Feature engineering (38 features): time cyclicals, 24h/168h lags, rolling stats, weather interactions, building metadata. Train/val split is time-based — last 3 months held out, no leakage.

For the optimization techniques required by the course we have: chunked CSV loading with `ThreadPoolExecutor` + `itertools.islice`, Numba `@njit(parallel=True)` kernels, Cython compiled kernels with typed memoryviews, CuPy GPU ops (falls back to CPU if no CUDA), PySpark distributed pipeline (falls back if Spark not installed), `ProcessPoolExecutor` for parallel per-site training (1.62x at 8 workers), and `scipy.optimize.minimize_scalar` for LR search.

## References

- Miller et al. (2020). The Building Data Genome Project 2. *Scientific Data.*
- Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS.*
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD.*
