# Real-World Project: Problem + Solution

## Primary Recommendation: Building Energy Consumption Prediction

### The Real-World Problem

**Who has the problem?**  
Building owners, facility managers, utilities, and sustainability teams.

**What is the problem?**  
Commercial buildings waste enormous amounts of energy because heating, cooling, and lighting are often run on fixed schedules or reactive rules instead of actual demand. That leads to:
- **High cost** – Energy is one of the largest operating expenses for buildings.
- **Carbon emissions** – Buildings account for a large share of global energy use and GHG emissions.
- **Comfort vs. efficiency trade-offs** – Without good predictions, operators either over-cool/over-heat (waste) or under-deliver (complaints).

**Why does prediction help?**  
If you can **predict next-hour or next-day energy use** accurately, you can:
- Pre-cool/pre-heat at the right times (demand response).
- Flag anomalies (e.g., a meter malfunction or a left-on system).
- Benchmark buildings and prioritize which ones to retrofit.
- Support sustainability and net-zero targets with data.

So the **real-world problem** is: **predict building energy consumption so that operators can optimize usage, cut cost, and reduce emissions.**

---

### The Data (Real, Large, Public)

**Dataset:** ASHRAE – Great Energy Predictor III (Kaggle)  
- **~53.6 million rows** of hourly meter readings.  
- **1,636 buildings**, **3,053 meters**, **19 sites** (North America & Europe).  
- **2 full years** (2016–2017).  
- Meter types: electricity, heating/cooling water, steam, solar, water/irrigation.  
- Includes building metadata (square footage, year built, primary use) and weather (air temperature).

**Why it fits the course:**  
Large scale (53M rows), time series, many buildings → need efficient loading (e.g., chunked/Dask), vectorized feature engineering, and possibly Numba for custom metrics or heavy loops.

---

### The Solution (What You Build)

**Goal:** Predict **meter-level energy consumption** (e.g., next hour or next 24 hours).

**High-level pipeline:**

1. **Data loading & cleaning**  
   - Load only needed columns and date ranges to keep memory manageable.  
   - Handle missing/misaligned timestamps and bad meter readings (e.g., negative or extreme values).  
   - Join meter → building metadata and weather.

2. **Feature engineering**  
   - Time: hour, day of week, month, holiday, weekend.  
   - Lag features: same hour yesterday, same day last week, rolling means.  
   - Building: size, primary use, year built.  
   - Weather: temperature (and optionally lagged/rolling).  
   - All implemented in a **vectorized** way (NumPy/pandas or Polars) so it scales.

3. **Train/validation split**  
   - Time-based split (e.g., last N months as validation) so the setup is realistic.

4. **Model(s)**  
   - Options: Gradient Boosting (XGBoost/LightGBM) or a simple LSTM/RNN for sequence modeling.  
   - Target: continuous energy consumption (regression).  
   - Metric: e.g. RMSE or CV-RMSE (coefficient of variation) per meter or per building type.

5. **Scalability & advanced Python**  
   - Process by building or by site in **parallel** (e.g., `multiprocessing` or `concurrent.futures`).  
   - Use **Dask** or chunked reads if you want to show “same pipeline on full 53M rows.”  
   - Use **Numba** for any custom cost or metric that has heavy loops.  
   - **Profiling** (e.g., `cProfile`, `memory_profiler`) to show bottlenecks and improvements.

**Deliverables:**  
- Reproducible pipeline (scripts or notebooks).  
- Clear metric (e.g., RMSE) and optional visualization (e.g., predicted vs actual for a few meters).  
- Short report (≤4 pages) explaining problem, data, method, and results.

---

### Why This Is a Strong Choice

| Criterion | How it fits |
|-----------|-------------|
| **Real-world** | Energy waste and carbon from buildings are a major, well-documented issue. |
| **Stakeholders** | Building operators, utilities, ESG/sustainability teams. |
| **Data** | One public Kaggle dataset, 53M rows, no scraping or APIs required. |
| **Feasibility** | You can subset by site or meter count and still show a complete solution; full scale is optional. |
| **Advanced Python** | Chunked I/O, Dask, vectorization, multiprocessing, Numba, profiling. |
| **Course alignment** | Data science + optimization at scale; clear story for proposal and final presentation. |

---

## Alternative 1: Store Sales / Demand Forecasting (Retail)

**Problem:** A grocery retailer (e.g., Corporación Favorita) needs to predict **unit sales per product per store** to reduce stockouts and food waste.

**Data:** Kaggle “Store Sales - Time Series Forecasting” (Favorita data).

**Solution:** Time series (or hierarchical) forecasting with features: date, store, product, promotions, holidays. Use vectorized feature engineering and, if needed, parallel runs per store or product family. Same advanced-Python angles: efficiency, scaling, profiling.

**Impact:** Less waste, better product availability.

---

## Alternative 2: Air Quality (PM2.5) Prediction

**Problem:** Cities and health agencies need **daily or sub-daily PM2.5 estimates** to issue advisories and protect vulnerable populations.

**Data:** NASA Airathon (Taipei, Delhi, LA) or Kaggle PM2.5 datasets (e.g., global 2010–2017 or Delhi).

**Solution:** Predict PM2.5 from satellite-derived AOD, weather, and optionally topography. Spatial–temporal modeling; possible use of Numba for custom spatial aggregates or distance-based features.

**Impact:** Public health, outdoor activity guidance, policy.

---

## Summary

- **Recommended main project:** **Building energy consumption prediction** with the ASHRAE dataset: clear real-world problem (cost + carbon), 53M-row dataset, and a concrete solution (forecasting pipeline + scalable, advanced Python).
- **Alternatives:** Retail demand forecasting (Favorita) or air quality (PM2.5) if you prefer a different domain; both are real-world and have public data and clear solutions.

If you tell me which option you prefer (ASHRAE vs. Favorita vs. PM2.5), the next step can be a minimal project layout (folder structure, one script for load + features + train) tailored to that choice.
