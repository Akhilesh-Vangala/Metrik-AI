# Metrik AI — Proposal Presentation Slides
## DS-GA 1019 Advanced Python for Data Science | Spring 2026

**Use this document to build your PPT or Google Slides.** Each section below = one slide. Copy the bullet points into your deck. Optional speaker notes are in *italics* at the end of each slide.

---

## SLIDE 1 — Title

**Metrik AI: Building Energy Consumption Prediction**

- A Real-World Project for Advanced Python for Data Science
- DS-GA 1019 | Spring 2026 | Project Proposal
- [Your name / Team name]

---

## SLIDE 2 — Real-World Motivation: Why This Project, Who It Addresses

**Why we are pursuing this project**

- **Buildings** are among the largest consumers of energy and a major source of GHG emissions globally.
- **Energy** is one of the largest operating expenses for commercial real estate—optimization directly reduces cost and carbon.
- **Operators** today often rely on fixed schedules and reactive rules instead of data-driven, demand-aware control.
- **Stakeholders we address:**
  - **Building owners & facility managers** — reduce cost and improve efficiency.
  - **Utilities** — demand response, grid stability.
  - **Sustainability / ESG teams** — data-backed progress toward net-zero and reporting.

*Speaker note: Frame this as “real-world impact”: cost, carbon, and operational decision-making.*

---

## SLIDE 3 — Problem Statement

**The problem**

- Commercial buildings **waste large amounts of energy** because heating, cooling, and lighting are run on fixed schedules or reactive rules rather than actual demand.
- Consequences:
  - **High cost** — Energy is one of the largest operating expenses for buildings.
  - **Carbon emissions** — Buildings account for a large share of global energy use and GHG emissions.
  - **Comfort vs. efficiency** — Without good predictions, operators either over-cool/over-heat (waste) or under-deliver (complaints).

**Problem statement (one sentence)**  
*Predict building energy consumption so that operators can optimize usage, cut cost, and reduce emissions.*

**What accurate prediction enables**
- Pre-cool/pre-heat at the right times (demand response).
- Flag anomalies (e.g., meter malfunction, equipment left on).
- Benchmark buildings and prioritize retrofits.
- Support sustainability and net-zero targets with data.

---

## SLIDE 4 — Solution: Approach and Expected Output

**What we propose**

We deliver **three components** that map directly to how this problem is solved in practice:

| Component | Approach | Output |
|-----------|----------|--------|
| **1. Forecasting** | Next-hour meter-level consumption; out-of-core, leakage-safe features; baseline (e.g. per-meter mean) vs. LightGBM; time-based train/validation split. | Point forecasts of energy use (e.g. next hour). |
| **2. Anomaly detection** | Residual-based: actual − predicted; scoring (e.g. z-score or MAD) over large arrays; Numba or vectorized implementation. | Per-meter (or per-building) anomaly scores to flag malfunction or abnormal use. |
| **3. Decision support** | Consumes forecasts and anomaly scores; ranks meters or buildings by priority. | Ranked list (e.g. top N meters/buildings to audit first). |

**High-level pipeline**  
Load data in chunks → merge metadata & weather → build leakage-safe features → train baseline + LightGBM → compute residuals → score anomalies → produce ranked audit list.

**Deliverables**
- Reproducible pipeline (scripts + config).
- Benchmark table (runtime, memory, speedup).
- Final report (≤4 pages) and code (CLI, tests, README).

---

## SLIDE 5 — Dataset: Overview, Scale, and Significance

**ASHRAE Great Energy Predictor III (GEPIII)**  
- **Source:** Kaggle — [ashrae-energy-prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction)  
- **Origin:** Building Data Genome Project 2; used for ASHRAE ML competition (Oct–Dec 2019) for long-term building energy prediction and measurement & verification (M&V).

**Scale and coverage**

| Aspect | Value |
|--------|--------|
| **Rows (train)** | ~53.6 million hourly meter readings |
| **Buildings** | 1,636 non-residential |
| **Meters** | 3,053 energy meters |
| **Sites** | 19 (North America & Europe) |
| **Time span** | 2 full years (2016–2017) |
| **Frequency** | Hourly |
| **Approx. size** | ~2–3 GB compressed (train + metadata + weather) |

**Significance**
- **Real buildings** — offices, retail, education, lodging; no synthetic data.
- **Public and free** — no scraping or external APIs; single download.
- **Scale** — large enough to require out-of-core processing, vectorization, and parallelization (directly aligned with course focus).

---

## SLIDE 6 — Dataset: Structure, Labels, Columns, and Features

**Main files**

| File | Role |
|-----|------|
| **train.csv** | Core time series: one row per (building, meter, timestamp); target = `meter_reading`. |
| **building_metadata.csv** | One row per building: site_id, primary_use, square_feet, year_built. |
| **weather_train.csv** | Site-level weather: site_id, timestamp, air_temperature, dew_temperature, cloud_coverage, precip_depth_1_hr. |

**train.csv columns**

| Column | Type | Description |
|--------|------|-------------|
| building_id | int | Links to building_metadata. |
| meter | int | Meter type code (see labels below). |
| timestamp | datetime | Start of hour (e.g. 2016-01-01 00:00:00). |
| meter_reading | float | **Target** — energy consumption for that hour (continuous, regression). |

**Meter type codes (labels)**

| Code | Meter type | Typical unit |
|------|------------|--------------|
| 0 | Electricity | kWh (Site 0: kBTU) |
| 1 | Chilled water | kWh-equivalent |
| 2 | Steam | kWh-equivalent |
| 3 | Hot water | kWh-equivalent |

**Features we will use (from metadata + weather + time)**  
Time: hour, day of week, month, is_weekend, holiday. Lags: same hour yesterday, same day last week. Rolling: 24h and 168h rolling mean (leakage-safe). Building: square_feet, primary_use, year_built. Weather: air_temperature (and optionally lagged). All built in a **vectorized, leakage-safe** way per chunk.

---

## SLIDE 7 — Dataset: Where It Was Used Before and Why It Matters

**Prior use and visibility**

- **ASHRAE Great Energy Predictor III competition** (Oct–Dec 2019) — goal: long-term prediction of building energy consumption for M&V and energy efficiency analysis.
- **Building Data Genome Project 2 (BDG2)** — dataset is part of this open project; described in **Miller et al. (2020), *Scientific Data* 7, 368** (see Literature Review slide).
- **Research and practice** — widely used for energy forecasting, anomaly detection, and benchmarking of scalable ML pipelines; cited in energy-informatics and building analytics work.

**Why this dataset fits our project**

- **No format lock-in** — we use a time-based split for validation (e.g. last 3–6 months) and can subset by site or rows for development.
- **Hierarchical structure** — site → building → meter; same site shares weather; supports partitioning and parallelization by site.
- **Regression target** — continuous `meter_reading`; metrics: RMSE or CV-RMSE; clear baseline (e.g. mean or “same hour yesterday”) and improvement target (e.g. 20–40% RMSE gain with LightGBM).

---

## SLIDE 8 — Potential Optimization Techniques and Approach

**How we will approach the problem (optimization and scalability)**

| Area | Technique | Our approach |
|------|-----------|--------------|
| **Data loading** | Out-of-core, chunked I/O | Never load full 53M rows; chunks of 1–2M rows; merge with in-memory metadata and weather; peak memory target &lt; 4 GB. |
| **Feature engineering** | Vectorization, leakage safety | NumPy/pandas per chunk; no row-wise Python loops; unit tests to ensure no future leakage. |
| **Modeling** | Baseline vs. strong model | Baseline: per-meter mean or “same hour yesterday”; stronger: LightGBM; time-based split; target: 20–40% RMSE improvement. |
| **Scalability** | Partitioning & parallelization | Partition by site (19) or time chunk; multiprocessing or concurrent.futures for feature build or per-site training; benchmark 1 vs. N workers (e.g. on 5M rows). |
| **Acceleration** | JIT compilation | Numba on hot paths (e.g. anomaly scoring over millions of residuals); compare runtime vs. pure Python loop; target ≥5× speedup where applicable. |
| **Profiling & benchmarking** | Identify bottlenecks | cProfile (CPU), memory_profiler (peak memory); benchmark table: baseline vs. optimized runtime, memory, speedup for load, features, anomaly. |

**Success targets (benchmark table)**  
- Load: chunked read completes, no OOM, peak &lt; 4 GB.  
- Features (5M rows): ≥2× speedup with vectorization (and more with parallel).  
- Forecasting: LightGBM ≥20% RMSE gain over baseline.  
- Anomaly (1M rows): ≥5× speedup with Numba or vectorization vs. naive loop.

---

## SLIDE 9 — How We Use What Is Taught in This Class (Advanced Python)

**Explicit mapping: course concepts → project**

| Course concept / technique | Where and why in our project |
|----------------------------|-----------------------------|
| **Chunked I/O & memory management** | Train table is never fully in memory; we stream 1–2M-row chunks and merge with metadata/weather. *Essential* for 53M rows on typical hardware. |
| **Vectorization (NumPy/pandas)** | All feature engineering is vectorized per chunk; no Python loops over rows. Enables scalable feature build and ≥2× speedup target. |
| **Multiprocessing / concurrent.futures** | Parallel feature build or per-site model training; each worker owns a partition, no shared mutable state. Demonstrates parallelization at scale. |
| **Numba JIT** | Applied to hot paths (e.g. anomaly scoring over millions of residuals). Shows when and how JIT yields measurable speedup (target ≥5× on 1M rows). |
| **Profiling (cProfile, memory_profiler)** | We identify bottlenecks (load, features, anomaly) and report runtime, peak memory, and speedup in a benchmark table. Directly ties course material to measurable optimization. |
| **Structured project: CLI, config, typing, tests** | Single entry point (e.g. `python -m src.cli run --config config.yaml`); YAML/TOML config for paths, chunk_size, seed; type hints on public APIs; pytest (leakage check + integration test). Ensures reproducibility and professional delivery. |

**Why this matters**
- At 53M rows, **naive Python** (full load, row-wise loops) would OOM or be impractically slow. The techniques from this class are **necessary** to deliver a working, scalable pipeline and to demonstrate advanced Python at scale.

---

## SLIDE 10 — Literature Review (If Required)

**Key references**

1. **Miller, C., et al. (2020).** *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition.* Scientific Data 7, 368.  
   - Describes the dataset, building stock, and meter types; standard citation for ASHRAE GEPIII data.  
   - arXiv: https://arxiv.org/abs/2006.02273  

2. **ASHRAE Great Energy Predictor III competition (2019).**  
   - Goal: long-term prediction of building energy consumption for measurement and verification (M&V) and energy efficiency analysis.  
   - Overview and results discussed in follow-up work (e.g. arXiv:2007.06933).

3. **Relevance to our project**  
   - Establishes that the dataset is real, non-residential, and intended for energy prediction and M&V—aligning with our forecasting, anomaly detection, and decision-support framing.

*Speaker note: If the instructor does not require literature review, this can be a single short slide or folded into the “Dataset: where used before” slide.*

---

## SLIDE 11 — Summary and Next Steps

**Summary**

- **Problem:** Building energy waste (cost, carbon); we predict consumption so operators can optimize.
- **Solution:** Three components — forecasting (baseline + LightGBM), anomaly detection (residual-based), decision support (ranked audit list).
- **Data:** ASHRAE GEPIII — 53.6M rows, 1,636 buildings, 3,053 meters, 19 sites, 2 years; public, no APIs.
- **Optimization & class alignment:** Chunked I/O, vectorization, multiprocessing, Numba, profiling, CLI/config/typing/tests — all directly applied at scale.

**Next steps (staged development)**

- **Phase 1:** Single-site prototype (chunked I/O, features, baseline + LightGBM).  
- **Phase 2:** Profiling and optimization targets.  
- **Phase 3:** Anomaly scoring and audit ranking; populate benchmark table.  
- **Phase 4:** Parallel scaling; optional full 53M run.  
- **Phase 5:** Final report (≤4 pages), code, and presentation.

**Initial results**  
None yet; we will report results in the final presentation and report.

---

**End of slide content.**  
*Total: 11 slides. Adjust timing to ~30–45 seconds per slide for a 5–6 minute talk; use speaker notes to stay on message.*
