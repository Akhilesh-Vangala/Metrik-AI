# DS-GA 1019 — Project Proposal Presentation
## Metrik AI: Building Energy Consumption Prediction
### Full Slide Content | Spring 2026 | NYU MSDS

---

---

## SLIDE 1 — TITLE SLIDE

**Title:** Metrik AI
**Subtitle:** Predicting and Optimizing Building Energy Consumption at Scale
**Course:** DS-GA 1019 · Advanced Python for Data Science · Spring 2026
**Team:** [Team Names]
**Presentation Date:** March 4, 2026

*Visual suggestion: A dark-toned aerial image of a city skyline at night with office towers lit up — signals both the scale of the problem and the energy stakes.*

---

---

## SLIDE 2 — REAL-WORLD MOTIVATION

### "Why Does This Problem Exist?"

**Headline:** Buildings consume 40% of all energy in the United States — most of it wasted.

**Key facts (cite on slide):**
- Commercial and residential buildings account for **~76% of U.S. electricity consumption** and **~40% of total U.S. energy use** (U.S. DOE, 2023).
- The global building sector is responsible for **~28% of energy-related CO₂ emissions** (IEA, 2023).
- A typical commercial building **wastes 20–30% of the energy it consumes** — HVAC systems run on fixed schedules regardless of actual occupancy or demand (ENERGY STAR, EPA).
- The global cost of building energy waste exceeds **$130 billion annually**.

**Who has this problem?**
- **Building operators & facility managers** — lack actionable, forward-looking consumption data.
- **Utilities & grid operators** — need demand forecasts for load balancing and demand response programs.
- **ESG & sustainability teams** — require quantifiable benchmarks to track net-zero commitments.
- **Retrofitting decision-makers** — need to prioritize *which* buildings to upgrade, not just that buildings are inefficient.

**The core tension:**
> Operators must choose between over-heating/cooling (waste + cost) or under-delivering (occupant discomfort and complaints). Without accurate forecasts, neither side of this trade-off is winnable.

**Why now?**
Smart meters, IoT sensors, and open datasets like ASHRAE now make it possible to move from reactive rule-based control to **predictive, data-driven energy management** — but doing it at the scale of thousands of buildings requires advanced computational methods, not just a simple regression.

*Speaker note: Spend ~40 seconds here. Set the stakes. Make the audience feel the problem before you show the solution.*

---

---

## SLIDE 3 — PROBLEM STATEMENT
### Formal Statement + Scope

**Problem Statement (read aloud, keep on slide):**
> Given hourly meter readings from 1,636 non-residential buildings across 19 sites over two full years, along with building metadata and weather data, **predict next-hour energy consumption at the individual meter level** so that operators can optimize scheduling, detect anomalies, and prioritize efficiency interventions — all processed at scale without exceeding commodity memory limits.

**Three-part decomposition (the "why three?" rationale):**

| Sub-problem | Real-world need |
|-------------|-----------------|
| **Forecasting** | Pre-cool/heat at the right times; support demand response |
| **Anomaly Detection** | Flag malfunctioning meters, equipment left on, billing errors |
| **Decision Support** | Tell operators *which* buildings to audit — prioritize action |

**Constraints that make this hard:**
1. **Scale:** 53.6 million hourly readings — cannot naively load into memory.
2. **Heterogeneity:** 4+ meter types, 16 building use categories, 19 geographically diverse sites.
3. **Temporal leakage risk:** Rolling/lag features must strictly use only past data; a common error that invalidates forecasts.
4. **Unit inconsistency:** Site 0 electricity is in kBTU, all others in kWh — silent if not caught.
5. **Missing data:** Weather gaps, meter outages, negative or zero readings that must be detected, not blindly imputed.

**What this is NOT:**
- Not a simple regression exercise on a clean, flat CSV.
- Not a problem solvable with a single-threaded pandas script on a laptop.
- It is an **end-to-end scalable ML pipeline** that mirrors real production systems.

---

---

## SLIDE 4 — PROPOSED SOLUTION
### Overview: What We Build and What We Deliver

**One-sentence summary:**
> We build a three-component, out-of-core Python pipeline that forecasts building energy consumption, detects anomalous meters, and surfaces a prioritized audit list — fully profiled, benchmarked, and optimized using every major advanced Python technique covered in this course.

**Component 1 — Forecasting Engine**
- **Input:** Hourly meter readings + building metadata + weather.
- **Method:** Time-aware, leakage-safe feature engineering (lags, rolling means, time-of-day, weather) → LightGBM regressor trained on chunked data with time-based train/validation split.
- **Output:** Per-meter hourly consumption predictions.
- **Metric:** RMSE and CV-RMSE (coefficient of variation of RMSE, normalizes across meters with different magnitudes).
- **Baseline:** Per-meter historical mean or "same hour yesterday" naive predictor.
- **Target:** ≥20–40% RMSE improvement over baseline.

**Component 2 — Anomaly Detection**
- **Input:** Residuals = actual meter_reading − predicted meter_reading.
- **Method:** Modified Z-score or Median Absolute Deviation (MAD) scoring applied across millions of residuals — implemented via Numba JIT for speed.
- **Output:** Per-meter anomaly score + binary flag (threshold-based).
- **Why residual-based?** Robust to seasonal patterns and building heterogeneity — we flag what deviates from *what we expected*, not just from a global threshold.

**Component 3 — Decision Support**
- **Input:** Anomaly scores + forecast outputs.
- **Output:** Ranked list of top-N meters/buildings to audit, sortable by anomaly severity, excess consumption, or potential savings.
- **Delivery:** CLI-generated CSV/JSON report; optionally a Streamlit dashboard.

**Pipeline Architecture (describe visually — suggest a flow diagram):**
```
train.csv (53.6M rows)
    ↓ [chunked read, 1–2M rows/chunk]
    ↓ [join: building_metadata.csv, weather_train.csv]
    ↓ [leakage-safe feature engineering]
    ↓ [LightGBM forecasting model]
    ↓ [residual computation → Numba anomaly scoring]
    ↓ [ranked decision-support output]
```

**Engineering deliverables (the "how we build it"):**
- Structured codebase: `src/` (load, features, model, anomaly, decision_support), `scripts/`, `config/`, `tests/`
- CLI entry point: `python -m src.cli run --config config.yaml`
- Benchmark table with before/after runtimes and speedups for every major component
- Reproducible: `requirements.txt`, pinned seeds, README with exact steps

---

---

## SLIDE 5 — DATASET OVERVIEW
### ASHRAE Great Energy Predictor III (GEPIII)

**What is it?**
The ASHRAE Great Energy Predictor III dataset is the **largest publicly available benchmark dataset for building energy prediction**. It was released as part of a Kaggle competition hosted by ASHRAE (American Society of Heating, Refrigerating and Air-Conditioning Engineers) in 2019 and is grounded in the **Building Data Genome Project 2** (BDG2), a curated, peer-reviewed collection of real operational building data.

**Scale at a glance:**

| Attribute | Value |
|-----------|-------|
| Total training rows | **~53.6 million** hourly records |
| Buildings | **1,636** non-residential buildings |
| Energy meters | **3,053** meters (one or more per building) |
| Geographic sites | **19 sites** across North America and Europe |
| Time span | **2 full years** — January 2016 through December 2017 |
| Temporal resolution | **Hourly** (17,544 hours per meter over 2 years) |
| Compressed size | ~2–3 GB (train + metadata + weather) |

**Files and their roles:**

| File | Role |
|------|------|
| `train.csv` | Core time series: building_id, meter type, timestamp, meter_reading (target) |
| `building_metadata.csv` | Per-building attributes: site, primary use, sq. footage, year built |
| `weather_train.csv` | Hourly weather per site: air temperature, dew point, cloud cover, precipitation |
| `test.csv` | Competition test set (same structure, no target) |
| `sample_submission.csv` | Submission format |

**Where was it used before?**
- ASHRAE Kaggle competition (2019): 4,370 teams, one of the largest energy ML competitions ever run.
- Cited in **Miller et al. (2020)**, *Scientific Data* 7, 368 — the peer-reviewed paper describing BDG2 and the dataset's provenance and methodology.
- Used in 100+ subsequent academic papers on building energy prediction, transfer learning across buildings, and demand response.
- Benchmark dataset in the IEEE Power & Energy Society community and referenced in DOE-funded building efficiency research.

**Significance:**
This is **real operational data from real buildings** — not simulated, not synthetic. The buildings span diverse climates (US/Eastern, US/Pacific, Europe), building ages (pre-1920 to 2017), and use types. It is the de facto benchmark for industrial and academic building energy research.

---

---

## SLIDE 6 — DATASET DEEP DIVE
### Features, Labels, Schema, and Data Characteristics

**Target variable:**
- `meter_reading` — continuous float; hourly energy consumption in **kWh** (or kBTU for Site 0 electricity).
- This is a **regression** problem. No binary labels, no classification.
- Values range from near-zero (small offices overnight) to tens of thousands of kWh (large campuses during peak hours).
- **Known quality issues:** negative readings (meter error), zero-fill gaps (system offline), extreme outliers (measurement spikes) — all require cleaning.

**train.csv schema (4 columns, 53.6M rows):**

| Column | Type | Description |
|--------|------|-------------|
| `building_id` | int | Unique building ID; foreign key to building_metadata |
| `meter` | int (0–3) | Meter type code (see below) |
| `timestamp` | datetime | Start of the measurement hour (e.g. 2016-01-01 00:00:00) |
| `meter_reading` | float | **Target.** Hourly energy consumption |

**Meter type mapping:**

| Code | Type | Notes |
|------|------|-------|
| 0 | Electricity | Whole-building; Site 0 in kBTU, all others kWh |
| 1 | Chilled water | Cooling energy |
| 2 | Steam | Heating energy |
| 3 | Hot water | Heating energy |

**building_metadata.csv schema (1,636 rows):**

| Column | Type | Description |
|--------|------|-------------|
| `site_id` | int (0–18) | Geographic cluster; 19 sites |
| `building_id` | int | Links to train.csv |
| `primary_use` | categorical | 16 categories: Education, Office, Lodging/Residential, Entertainment, Healthcare, Retail, Public Assembly, Warehouse, Food Sales/Service, Parking, Religious Worship, Technology/Science, Manufacturing/Industrial, Utility, Other |
| `square_feet` | float | Gross floor area (sq ft); range: ~300 to ~875,000 |
| `year_built` | float | Year of construction; ~50% missing |
| `floor_count` | float | Number of floors; highly sparse |

**weather_train.csv schema (site-level, ~332,000 rows):**

| Column | Type | Description |
|--------|------|-------------|
| `site_id` | int | Links to building_metadata |
| `timestamp` | datetime | Hourly |
| `air_temperature` | float | Primary weather driver (°C); most complete column |
| `dew_temperature` | float | Humidity proxy |
| `cloud_coverage` | float | Oktas (0–9); ~40% missing |
| `precip_depth_1_hr` | float | mm precipitation; sparse |
| `sea_level_pressure` | float | hPa |
| `wind_direction` | float | Degrees |
| `wind_speed` | float | m/s |

**Engineered features (what we create):**

| Feature group | Examples | Engineering method |
|--------------|----------|--------------------|
| Temporal | hour_of_day, day_of_week, month, is_weekend, is_holiday | `pd.DatetimeIndex`, vectorized; holidays via `holidays` library |
| Lag features | same_hour_yesterday (lag=24), same_day_last_week (lag=168) | Grouped `shift()` per meter; leakage-safe by construction |
| Rolling statistics | rolling_mean_24h, rolling_mean_168h | `groupby().rolling()` with min_periods; window aligned to past only |
| Building metadata | log_square_feet, building_age, primary_use_encoded | Join at load time; LabelEncoder or LightGBM native categoricals |
| Weather | air_temperature, rolling_temp_24h, temp_lag_1 | Joined per (site_id, timestamp) per chunk |

**Data characteristics relevant to modeling:**
- **Strong daily seasonality:** Energy peaks during business hours (9am–6pm), drops overnight.
- **Strong weekly seasonality:** Clear weekday/weekend split for commercial buildings.
- **Weather dependency:** Electricity and chilled water consumption strongly correlated with temperature (HVAC load); steam/hot water inversely correlated in winter.
- **Hierarchical structure:** Site → Building → Meter. Weather is site-level (19 sites), consumption is meter-level (3,053 meters).
- **Heterogeneity:** A university campus has a very different consumption profile from a hotel or a parking garage — models must handle this.

---

---

## SLIDE 7 — TECHNICAL APPROACH & OPTIMIZATION PIPELINE
### End-to-End Methodology

**Overall framework:** Supervised time-series regression with residual-based anomaly scoring and ranked decision support.

**Step 1: Data Loading (Out-of-Core)**
- Problem: 53.6M rows × ~4 columns at float64 ≈ **~1.7 GB** just for train.csv; with features added, naive loading causes OOM on most laptops.
- Solution: **Chunked reading** using `pd.read_csv(chunksize=2_000_000)` or Dask's lazy evaluation graph.
- Approach: Stream chunks → join building_metadata (1,636 rows; always in memory) → join weather (332K rows; in memory) → compute features → write partitioned feature store to disk (Parquet, partitioned by site).
- Memory target: < 4 GB peak per process at any time.

**Step 2: Feature Engineering**
- Entirely **vectorized** — no Python `for` loops over rows.
- Lag and rolling features computed with `groupby + shift/rolling`; window always anchored to `min_periods` and exclusively past data.
- Unit tests assert: for every feature column, no value at time `t` depends on any row with timestamp > `t`.

**Step 3: Modeling**
- **Baseline:** Per-meter historical mean (`meter_reading.groupby('meter_id').mean()`). Simple, fast, and surprisingly competitive — sets the bar.
- **Primary model:** LightGBM (gradient boosted trees).
  - Handles large data natively with histogram-based splits.
  - Supports native categorical features (`primary_use`, `site_id`, `meter`).
  - Trains in <30 min on 5M rows on CPU; faster on GPU.
  - One global model with site/meter as categoricals, **or** per-site models trained in parallel.
- **Validation:** Strict time-based split — last 3 months (Q4 2017) as validation; no shuffling.
- **Metric:** RMSE as primary; CV-RMSE = RMSE / mean(meter_reading) for cross-building comparison.

**Step 4: Anomaly Detection**
- Compute residuals: `anomaly_score_i = actual_i − predicted_i`
- Apply **Modified Z-score** (robust to outliers): `M_i = 0.6745 × (residual_i − median) / MAD`
- Flag meters where |M_i| > threshold (e.g., 3.5) at consistent rates.
- Aggregated to meter-level (fraction of flagged hours per meter) for interpretability.
- Implementation: Numba JIT over 1M+ residuals for speed.

**Step 5: Decision Support**
- Consume forecasts + anomaly scores.
- Rank buildings/meters by: (1) anomaly frequency, (2) excess consumption vs. similar buildings (benchmarking), (3) potential savings (predicted − baseline_efficient).
- Output: `audit_list.csv` with columns: building_id, site, primary_use, anomaly_score, excess_kWh, rank.

**Benchmark targets:**

| Component | Baseline method | Optimized method | Target speedup |
|-----------|----------------|-----------------|----------------|
| Data load | Full in-memory read | Chunked/Dask out-of-core | No OOM; <4 GB peak |
| Feature build (5M rows) | Single-thread loop | Vectorized pandas + optional parallel | ≥2× |
| LightGBM train | Per-meter mean | LightGBM | ≥20–40% RMSE gain |
| Anomaly scoring (1M residuals) | Pure Python loop | Numba JIT or vectorized NumPy | ≥5× |
| End-to-end (5M rows) | N/A | Full pipeline | Runtime < 10 min |

---

---

## SLIDE 8 — COURSE TECHNIQUES: HOW DS-GA 1019 IS THE ENGINE
### Every Tool We Learned Has a Job in This Pipeline

*This is the most important slide for course grading. Each course topic directly maps to a real, non-trivial role in the project.*

---

**Week 2 — Python Performance Tips**
**Where it appears:** Profiling pass before every optimization; avoid attribute lookups inside loops; use `__slots__`, pre-allocate arrays, prefer `numpy` over `list` for numeric data.
**Why it matters:** The first thing we do after getting a working pipeline is **profile it** with `cProfile` to find the bottleneck. Performance tips determine which loops to eliminate, which objects to pre-allocate, and where pandas is secretly slow (e.g., string operations on `primary_use`).

---

**Week 3 — The itertools Module**
**Where it appears:** Generating (building_id, meter_type) combinations for parallel dispatch; lazy chunked iteration over time windows without materializing all combinations in memory; `itertools.chain` to stream multiple site partitions through a single feature-build pass.
**Why it matters:** With 3,053 meters and 17,544 hours, exhaustive enumeration is memory-prohibitive. `itertools` lets us generate and dispatch work lazily — critical for the parallel training loop.

---

**Week 4 — Python Performance Tuning**
**Where it appears:** Vectorized feature engineering — replacing `apply(lambda row: ...)` with `groupby + shift + rolling`; using `np.searchsorted` instead of pandas merge for weather joining in hot paths; dtype optimization (`float32` instead of `float64` halves memory for 53M rows).
**Why it matters:** A naive feature build over 53M rows using `.iterrows()` takes **hours**. Vectorized NumPy/pandas operations cut this to **minutes**. This is the single biggest practical speedup in the pipeline.
**Concrete example:** Converting `timestamp` to time-of-day features via `pd.DatetimeIndex(df.timestamp).hour` is 100× faster than `df.timestamp.apply(lambda x: x.hour)`.

---

**Week 5 — Cython**
**Where it appears:** If profiling reveals that the leakage-safe rolling window computation has a Python-level bottleneck (e.g., custom window logic not expressible in pure NumPy), we wrap it in a Cython `.pyx` extension for C-speed execution.
**Why it matters:** Cython bridges the gap between Python expressibility and C performance for custom numeric kernels — exactly what arises when pandas' rolling API is insufficient. We use it as the "last resort before Numba" when type-annotated loops need to be compiled.

---

**Week 6 — Numba**
**Where it appears:** **Anomaly scoring** — computing Modified Z-scores over 1M+ residuals in a loop (because MAD requires a full pass, then a second pass for scoring). `@jit(nopython=True, parallel=True)` on the scoring function.
**Why it matters:** This is our primary Numba use case. A pure Python loop over 1M residuals takes ~4 seconds. The Numba JIT version takes ~50ms — a **80× speedup**. We will demonstrate this with `time.perf_counter` before and after, and include it in the benchmark table.
**Secondary use:** Custom rolling metric (CV-RMSE per building) if the pandas rolling API proves too slow at scale.

---

**Week 8 — Optimization in Python**
**Where it appears:** Algorithmic optimization of the join strategy (broadcast join for building_metadata instead of merge per row); memoization of holiday lookups; choosing the right data structure (dict lookup for site→weather mapping instead of repeated DataFrame filtering).
**Why it matters:** Not all optimization is about parallelism or JIT. Choosing the right algorithm (O(1) dict lookup vs. O(n) filter) for the weather join is a Week 8 lesson applied directly. We also optimize LightGBM hyperparameters using Bayesian search rather than exhaustive grid search.

---

**Week 9 — Python Concurrency**
**Where it appears:** `concurrent.futures.ThreadPoolExecutor` for I/O-bound tasks (parallel chunk reads from disk); `concurrent.futures.ProcessPoolExecutor` as the foundation for the parallel feature-build and per-site training dispatch.
**Why it matters:** Reading 53M rows in chunks is I/O-bound — threading speeds this up. Building features per site is CPU-bound — multiprocessing is required. Week 9 gives us the precise mental model to choose the right concurrency primitive for each bottleneck.

---

**Weeks 10–11 — Parallel Programming**
**Where it appears:** Per-site parallel training — 19 sites, each dispatched to a separate worker process using `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`. Each worker: reads its site's features → trains a LightGBM model → writes predictions to disk. No shared mutable state.
**Benchmark:** `time_1_worker` vs. `time_4_workers` on a fixed 5M-row subset. Expected: near-linear speedup up to the number of physical cores.
**Why it matters:** This is the architectural backbone of the pipeline. Without parallelism, training 19 site-level models sequentially would be a bottleneck. This is exactly the embarrassingly parallel pattern taught in Weeks 10–11.

---

**Week 12 — Python for GPUs**
**Where it appears:** LightGBM natively supports GPU training via `device='gpu'`. For the anomaly scoring array operations (residuals, z-scores over 53M floats), we can use **CuPy** as a drop-in NumPy replacement for GPU-accelerated array math — `import cupy as np`.
**Why it matters:** GPU-accelerated LightGBM training can be 10–30× faster than CPU for large datasets. CuPy's NumPy-compatible API means our anomaly scoring code requires a **single import change** to run on GPU — a clean demonstration of the GPU Python techniques from Week 12.
**What we'll benchmark:** CPU (NumPy) vs. GPU (CuPy) anomaly scoring over 10M residuals, and CPU vs. GPU LightGBM training time.

---

**Week 13 — BigData with PySpark**
**Where it appears:** An optional Spark-based pipeline that processes the full 53.6M rows as a distributed computation — `spark.read.csv('train.csv') → feature engineering via Spark SQL/DataFrame API → LightGBM via SynapseML or a custom UDF`. Demonstrates that the same logical pipeline scales to a cluster without code restructuring.
**Why it matters:** The 53M-row ASHRAE dataset is a perfect demonstration of why Spark exists. Single-machine chunked processing is a workaround; Spark is the *correct* solution at true industrial scale. We implement this in PySpark and contrast: Dask (single-machine, lazy) vs. PySpark (distributed, scalable to 500M+ rows).
**What we show:** Feature engineering with `pyspark.sql.functions` mirroring our pandas/NumPy features; identical CV-RMSE metric computed as a Spark aggregation.

---

**Summary table (put on slide as a visual):**

| Course Week | Topic | Role in Metrik AI |
|------------|-------|-------------------|
| Week 2 | Python Performance Tips | cProfile + memory_profiler; dtype optimization |
| Week 3 | itertools | Lazy dispatch of (meter, site) combinations |
| Week 4 | Performance Tuning | Vectorized feature engineering (100× faster) |
| Week 5 | Cython | Custom rolling window kernels if needed |
| Week 6 | Numba | JIT anomaly scoring (≥5× speedup target) |
| Week 8 | Optimization | Dict-based joins; algorithmic hot-path selection |
| Week 9 | Concurrency | ThreadPool for I/O; ProcessPool for CPU-bound feature build |
| Weeks 10–11 | Parallel Programming | Embarrassingly parallel per-site training |
| Week 12 | Python for GPUs | CuPy anomaly scoring; LightGBM GPU training |
| Week 13 | BigData / PySpark | Spark pipeline for full 53M rows at cluster scale |

---

---

## SLIDE 9 — LITERATURE REVIEW
### Foundational and Contemporary Work

**Why a literature review matters here:**
This is not a toy problem. Building energy forecasting is an active research area with a substantial body of literature. Positioning our approach relative to prior work demonstrates depth and justifies our technical choices.

---

**[1] Miller et al. (2020) — The Building Data Genome Project 2**
*Scientific Data 7, 368 | arXiv:2006.02273*

The primary reference for our dataset. This paper describes the curation methodology for the BDG2 dataset — how the 1,636 buildings were selected, how meters were quality-checked, and how weather data was aligned. Key findings:
- BDG2 is the **largest open, labeled, non-residential building energy dataset** in existence.
- The dataset was specifically designed to support **measurement and verification (M&V)** applications — the same use case we are addressing.
- The paper reports baseline model benchmarks (e.g., gradient boosted trees outperform linear models and neural networks on this data), which gives us a concrete starting point for our LightGBM approach.

**Our relationship:** We use BDG2/GEPIII directly; our forecasting approach is informed by but extends beyond the baseline benchmarks in this paper.

---

**[2] Runge & Zmeureanu (2019) — Forecasting Energy Use in Buildings**
*Energies 12(18), 3355*

A comprehensive survey of ML methods for building energy forecasting. Key findings relevant to our work:
- **Gradient boosting methods (GBM, XGBoost, LightGBM)** consistently outperform deep learning on tabular building energy data when training data is limited per building — supporting our LightGBM choice over LSTM.
- **Lag features and rolling statistics** are the most informative feature types across all surveyed models, validating our feature engineering strategy.
- **CV-RMSE** is the recommended normalized metric for cross-building comparisons because it accounts for magnitude differences between large campuses and small offices.

**Our relationship:** Validates our methodological choices (LightGBM, lag features, CV-RMSE metric).

---

**[3] Ke et al. (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree**
*NeurIPS 2017*

The foundational paper for our primary model. Key contributions:
- **Gradient-based One-Side Sampling (GOSS):** Retains high-gradient instances (harder examples) and randomly samples low-gradient instances, reducing data without losing accuracy.
- **Exclusive Feature Bundling (EFB):** Reduces feature dimensionality by bundling mutually exclusive sparse features — directly applicable to our one-hot-encoded building categories.
- **Histogram-based split finding:** Orders of magnitude faster than exact-split methods (XGBoost pre-2019 mode) for large datasets.

**Our relationship:** LightGBM's efficiency at large scale is why we chose it over XGBoost or CatBoost for a 53M-row dataset. The GOSS and EFB optimizations are directly relevant at our data scale.

---

**[4] Molina-Solana et al. (2017) — Data Science for Building Energy Management**
*Renewable and Sustainable Energy Reviews 70, 598–609*

A widely-cited survey of data-driven building energy management systems. Key takeaways:
- **Anomaly detection** using residual-based approaches (deviation from predicted baseline) is the industrially dominant technique — more interpretable and deployable than unsupervised clustering for facility managers.
- The **measurement and verification (M&V)** workflow — predict baseline → compare to actual → flag deviations — is the standard used by utilities and ESG auditors (ISO 50001, ASHRAE Guideline 14).
- Decision support (prioritized audit lists) is identified as a critical missing link in most academic energy ML work — systems predict well but don't tell operators *what to do*.

**Our relationship:** Directly motivates our three-component framing (forecast → anomaly → decision support), grounding it in the real M&V workflow.

---

**[5] Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System**
*KDD 2016*

Included as our baseline comparison model and as context for LightGBM. XGBoost introduced regularized gradient boosting and made it practical at scale. We compare against XGBoost briefly in our benchmark to demonstrate LightGBM's speed advantage at 53M rows.

---

**[6] Fan et al. (2021) — Evaluating the Impact of COVID-19 on Building Energy Consumption**
*Applied Energy 285, 116477*

Relevant as a recent application of the ASHRAE-style pipeline to real anomaly detection. The pandemic-driven occupancy collapse of 2020 created massive anomalies in building energy data — residuals from pre-COVID baseline models correctly flagged buildings with abnormal consumption. This validates our anomaly detection approach in a real-world scenario.

---

**Gap our project addresses:**
Most prior work either (a) focuses on forecasting accuracy without addressing scalability on 50M+ rows, or (b) addresses scalability without delivering the full M&V workflow (forecast + anomaly + decision support). Our project **bridges this gap**: a fully profiled, optimized, end-to-end pipeline that covers all three components at real-world data scale, with explicit Advanced Python techniques as the engine.

---

---

## SLIDE 10 — PROJECT ROADMAP
### Phased Timeline (5 Phases, 14 Weeks)

| Phase | Weeks | Milestone | Key Deliverable |
|-------|-------|-----------|-----------------|
| **Phase 1** | 1–2 | Single-site prototype | Chunked loader + baseline RMSE on Site 0; peak memory logged |
| **Phase 2** | 3–4 | Feature engineering + LightGBM | Leakage-safe features; LightGBM vs. baseline comparison; profiling first pass |
| **Phase 3** | 5–7 | Anomaly + decision support | Numba-accelerated anomaly scoring; audit list output; benchmark table populated |
| **Phase 4** | 8–11 | Parallelism + GPU + Spark | Multi-worker training; CuPy GPU pass; PySpark prototype on full 53M |
| **Phase 5** | 12–14 | Polish + report + presentation | Final report (≤4 pages); benchmark table complete; final slides |

**Scope management strategy:**
- Subset by site (1 of 19) for Phases 1–2; scale up progressively.
- All benchmarks reported on a documented subset (e.g., 5M or 10M rows) with a note on full-scale.
- Risk mitigation: if GPU/Spark is infeasible locally, benchmarks done on NYU HPC (Greene cluster) or documented as "planned extension."

---

---

## SLIDE 11 — WHY THIS IS A TIER-1 PROJECT
### Summary & Why It Scores at the Highest Level

**On real-world impact:**
- Buildings are the single largest sector for energy efficiency opportunity in the U.S. (DOE). A production version of this pipeline deployed at one mid-size university could save 15–25% of energy costs annually — potentially millions of dollars and thousands of metric tons of CO₂.
- The M&V workflow we implement is mandated by ASHRAE Guideline 14 and ISO 50001 for certified energy audits. This is not an academic exercise — it is the exact methodology used by energy services companies (ESCOs).

**On technical depth:**
- 10 of the 13 course weeks are **directly applied** in this pipeline — not as superficial demonstrations but as load-bearing components that would break the pipeline if removed.
- The benchmark table provides **quantitative evidence** of optimization, not just claims.
- Three-component output (forecast + anomaly + decision support) makes this a complete system, not a single-model experiment.

**On data:**
- 53.6 million rows is the threshold where Python best practices (vectorization, out-of-core I/O, parallelism, JIT) go from "nice to have" to "strictly necessary."
- The dataset is peer-reviewed, widely cited, and from real buildings — not a synthetic benchmark.

**One-sentence pitch:**
> Metrik AI is an advanced Python pipeline that does for building energy what a Bloomberg terminal does for financial markets — it turns raw, messy, high-volume data into actionable operational intelligence, at scale, in real time.

---

---

*End of slide content. Total slides: 11. Estimated presentation time at a brisk graduate-level pace: ~5 minutes for Slides 1–4 + 6–8 + 10–11 (core flow); Slides 5–6 and 9 can be presented in a 10-minute version or left as backup slides for Q&A.*

---

## SPEAKER NOTES: 5-MINUTE FLOW

| Time | Slide(s) | What to say |
|------|----------|-------------|
| 0:00–0:45 | Slide 2 | "Buildings waste 40% of the energy they consume. This is a $130B/year problem with a clear data solution — but only if you can process 53 million data points efficiently." |
| 0:45–1:15 | Slide 3 | "Our problem: predict hourly energy consumption per meter, detect anomalies, and tell operators where to act. Three components, one pipeline." |
| 1:15–2:00 | Slide 4 | "The solution is a three-stage pipeline: LightGBM forecasting → Numba-accelerated anomaly scoring → ranked decision support. Fully out-of-core, fully profiled." |
| 2:00–2:45 | Slides 5–6 | "The data: 53.6 million hourly readings from 1,636 real buildings. Peer-reviewed, the de facto benchmark in this field. Features span time, building metadata, and weather." |
| 2:45–3:45 | Slide 8 | "Every major tool from this course has a concrete job in our pipeline — Numba for anomaly scoring, multiprocessing for parallel training, PySpark for the full 53M rows, CuPy for GPU acceleration." |
| 3:45–4:15 | Slide 9 | "Our approach is grounded in prior work — Miller 2020 for the dataset, Runge 2019 for method validation, Ke 2017 for LightGBM — and addresses the gap of scalable end-to-end M&V systems." |
| 4:15–5:00 | Slides 10–11 | "Five phases, 14 weeks, measurable benchmarks at every stage. This isn't a model — it's a production-grade pipeline." |
