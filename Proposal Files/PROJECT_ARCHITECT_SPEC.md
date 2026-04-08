# DS-GA 1019 Project Architect Spec
## Building Energy Consumption Prediction (ASHRAE)

---

## A) Chosen Framing + Justification

**Chosen framing: B) Forecasting + Anomaly Detection + Decision Support**

### Five-bullet justification (why this scores highest for grading)

1. **Three distinct technical components** — Forecasting, anomaly detection, and decision support give three separate surfaces to demonstrate advanced Python (out-of-core forecasting pipeline, efficient anomaly scoring at scale, aggregation/reporting). A single “forecasting-only” pipeline is easier to dismiss as generic ML; this framing forces explicit scalability and tooling choices per component.

2. **Directly addresses “demonstrated understanding of the problem”** — The instructor wants to see that you understand the real-world use case. Operators need not only predictions but (a) which meters/buildings are behaving abnormally and (b) which to act on first. Forecasting + anomaly + decision support maps cleanly to: “predict usage → flag bad behavior → prioritize actions,” which is exactly how this problem is solved in practice.

3. **Maximizes “Advanced Python” artifacts** — You will deliver: chunked/parallel data loading, vectorized and leakage-safe feature engineering, a baseline vs. stronger model comparison, parallel partitioning (by site or time chunk), JIT acceleration (Numba) for hot paths (e.g., anomaly scoring), profiling/benchmarking with concrete speedup and memory numbers, and a small but structured codebase (CLI, config, typing, tests). That is a full checklist, not a single script.

4. **Feasible in one semester** — Forecasting remains the core (most effort); anomaly detection is residual-based (forecast vs. actual → score with a simple rule or Numba-accelerated batch); decision support is one script that consumes forecasts and anomaly outputs and produces a ranked list or report. No new datasets, no external APIs, no open-ended research.

5. **Presentation-ready in 5 minutes** — One minute: problem (waste, cost, carbon). One minute: we predict next-hour usage at scale. One minute: we flag anomalous meters/buildings. One minute: we support “which to audit first.” One minute: how we did it at 53M rows (out-of-core, parallel, Numba, benchmarks). Clear narrative with no scope creep.

---

## B) Milestone Plan (Week-by-Week)

Assume a 14-week semester; proposal presentation around Week 6–7.

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1–2 | Setup + data + baseline | Repo structure (src/, config/, scripts/, tests/). Data downloaded and validated. Chunked loader that streams train.csv; peak memory logged. Naive baseline (e.g., per-meter mean or last-hour) with a time-based split; baseline RMSE on a subset (e.g., 5M rows) documented. |
| 3 | Feature engineering (leakage-safe) | Feature module: time (hour, dow, month, weekend, holiday), lags (same hour yesterday, same day last week), rolling (24h, 168h) with correct expanding window. Unit tests that assert no future leakage. Vectorized implementation; benchmark: time and memory for 5M-row chunk. |
| 4 | Forecasting model + pipeline | LightGBM model trained on chunked or sampled data; time-based validation RMSE. Pipeline script that: load chunk → merge metadata/weather → features → predict. Comparison table: baseline vs. LightGBM (RMSE, training time). |
| 5 | Scalability + parallelization | Partitioning scheme (by site or by time chunk). Parallel feature build or parallel per-site training using multiprocessing or concurrent.futures. Benchmark: single-thread vs. N workers (runtime, speedup). |
| 6 | Proposal prep + presentation | Proposal presentation (5 min + Q&A). Slides: problem, data, three components (forecast, anomaly, decision), technical approach, benchmark targets. Proposal doc updated with chosen framing and Section 3 replacement. |
| 7–8 | Anomaly detection | Anomaly module: residuals (actual − predicted) from forecast pipeline; scoring (e.g., z-score or MAD) applied in a Numba-accelerated or vectorized way over large arrays. Output: per-meter (or per-building) anomaly score. Benchmark: runtime and memory for full 53M or 10M sample. |
| 9 | Decision support | Decision-support script: inputs = forecast outputs + anomaly scores; output = ranked list (e.g., “top N meters to audit” or “top N buildings by predicted savings”). Optional: simple CLI flag to export report (CSV/JSON). |
| 10 | Acceleration + profiling | Numba JIT on the hottest path (e.g., anomaly scoring or custom rolling). cProfile + memory_profiler runs; identify and fix one bottleneck. Benchmark table filled: baseline vs. optimized for load, features, anomaly. |
| 11 | Engineering polish | CLI entry point (e.g. `python -m src.cli run --config config.yaml`). Config file (paths, chunk_size, seed). Type hints on public APIs. Logging. requirements.txt + README. |
| 12 | Testing + reproducibility | Pytest: at least (1) feature tests (no future leakage), (2) one end-to-end run on tiny fixture. Document Python version and OS; fix random seed in config. |
| 13 | Report + slides | Final report (≤4 pages): problem, data, solution (three components), benchmark table, results, limitations. Final presentation (5 min + 1–2 min Q&A) rehearsed. |
| 14 | Submission | Final code, report, and slides submitted per course instructions. |

---

## C) Rewritten Section 3 (“The Solution”)

*Provided as a separate LaTeX file for drop-in replacement: `project_proposal_section3.tex`*

---

## D) Advanced Python Deliverables Checklist

Artifacts you must be able to submit or demonstrate (for “Advanced Python” grade):

| # | Artifact | Where it lives | What grader should see |
|---|----------|-----------------|-------------------------|
| 1 | Out-of-core / chunked data loading | `src/load.py` or equivalent | Train table never fully in memory; chunk size and peak memory documented (e.g., in benchmark table). |
| 2 | Leakage-safe, vectorized feature engineering | `src/features.py` | Time, lags, rolling features; unit test proving no future info in train features. |
| 3 | Baseline vs. stronger model comparison | Report + code | At least two models (e.g., per-meter mean vs. LightGBM); RMSE and runtime in table. |
| 4 | Parallelization (multiprocessing or concurrent.futures) | `src/` or `scripts/` | One of: parallel feature build, parallel per-site training, or parallel anomaly pass; speedup reported. |
| 5 | Numba (or Cython) acceleration | One or more modules | At least one hot path (e.g., anomaly scoring or custom rolling) JIT-compiled; before/after runtime in benchmark table. |
| 6 | Profiling and benchmarking | Scripts or notebook | cProfile and/or memory_profiler used; benchmark table with: component, baseline, optimized, runtime, peak memory, speedup. |
| 7 | Structured project and CLI | Repo root + `src/` | Package layout (src/, config/, scripts/, tests/); at least one CLI entry point (e.g. `python -m src.cli run`). |
| 8 | Config and reproducibility | `config/` or equivalent | Paths, chunk_size, seed in config; requirements.txt; README with run instructions. |
| 9 | Type hints and tests | Across `src/` + `tests/` | Type hints on public functions; ≥1 test for feature leakage, ≥1 integration test on small data. |
| 10 | Three-component outputs | Code + report | Forecasts (predictions), anomaly scores (e.g., per meter), and decision-support output (e.g., ranked audit list) all produced by the pipeline. |

---

## E) Risk Register (Top 8 Risks + Mitigation)

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | Full 53M-row pipeline too slow on laptop | High | High | From day one, support subset mode (e.g., one site or first N rows). Report benchmarks on 5–10M rows; state “full 53M run on cluster/larger machine” as optional. |
| 2 | Feature leakage invalidates results | Medium | High | Implement only past-based features; add unit test that checks no future timestamp in feature computation; document split (time-based) in report. |
| 3 | Anomaly detection scope creep (e.g., ML-based) | Medium | Medium | Define anomaly as residual-based only: actual − predicted, then scalar score (z-score/MAD). No unsupervised clustering or complex models unless time permits. |
| 4 | Numba/Cython gives little speedup | Medium | Low | Choose one clear hot path (e.g., scoring 10M residuals in a loop). If speedup is minimal, still document “attempted JIT; bottleneck was elsewhere” and show profiling evidence. |
| 5 | Instructor sees “generic ML project” | Medium | High | Emphasize in slides and report: (1) three components (forecast + anomaly + decision), (2) scale (53M rows, out-of-core, parallel), (3) benchmark table with speedup and memory, (4) engineering (CLI, config, tests). |
| 6 | Team understaffed or late start | Medium | High | Milestone plan front-loads baseline and features (Weeks 1–4); anomaly and decision support are thin layers. If behind, drop “full 53M” and deliver on 5–10M with clear documentation. |
| 7 | Data quality issues (missing weather, bad meters) | Low | Medium | Document missingness in EDA; drop or impute with simple rule (e.g., forward-fill weather); exclude meters with >X% missing in training. |
| 8 | Reproducibility failures on grader’s machine | Low | High | Pin key versions in requirements.txt; use relative paths and config for data dir; README with exact steps (install, download data, run CLI); optional Docker or environment.yml. |

---

## Benchmark Table Template (Measurable Success)

Use this table in your report. Fill with actual numbers from your runs.

| Component | Baseline | Optimized | Runtime (s) | Peak memory (GB) | Speedup | Notes |
|-----------|----------|-----------|-------------|-------------------|---------|-------|
| Data load | Naive (full read or OOM) | Chunked read | — | &lt; 4 | N/A (completes) | Target: no OOM; peak &lt; 4 GB. |
| Feature build (5M rows) | Single-thread pandas | Vectorized (+ optional parallel) | — | — | Target ≥ 2× | |
| Forecasting train | Per-meter mean | LightGBM | — | — | — | Target: RMSE improvement 20–40% vs baseline. |
| Anomaly scoring (1M rows) | Python loop | Numba or vectorized | — | — | Target ≥ 5× | |
| End-to-end (subset) | — | Full pipeline | — | — | — | e.g. 5M or 10M rows. |

**Success criteria (realistic):**
- **Load:** Chunked loader completes without OOM; peak memory under 4 GB for any single process.
- **Features:** Feature build on 5M-row sample at least 2× faster with vectorization (and optionally more with parallel).
- **Forecasting:** LightGBM RMSE at least 20% better than a simple baseline (e.g., mean or last-hour) on the same split.
- **Anomaly:** Anomaly pass on 1M residuals at least 5× faster with Numba or full vectorization than a naive Python loop.
- **Profiling:** At least one bottleneck identified and improved, with numbers in the table.

---

*End of Project Architect Spec*
