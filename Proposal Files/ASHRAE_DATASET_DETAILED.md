# ASHRAE Great Energy Predictor III — Dataset Link & Full Description

## Dataset link (free access)

**Official competition & data:**  
**https://www.kaggle.com/competitions/ashrae-energy-prediction**

- **Data tab (download):** https://www.kaggle.com/competitions/ashrae-energy-prediction/data  
- **Access:** Free. Requires a free Kaggle account and accepting the competition rules (“Join competition”).  
- **Download:** Via “Download All” on the data page, or via Kaggle API:  
  `kaggle competitions download -c ashrae-energy-prediction -p data/`

---

## 1. Overview

The **ASHRAE Great Energy Predictor III (GEPIII)** dataset was used for a machine learning competition run by ASHRAE (American Society of Heating, Refrigerating and Air-Conditioning Engineers) in October–December 2019. The goal was **long-term prediction of building energy consumption** for measurement and verification (M&V) and energy efficiency analysis.

The data are part of the **Building Data Genome Project 2** and are described in:

- *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition* (Miller et al., 2020), Scientific Data 7, 368.  
- arXiv: https://arxiv.org/abs/2006.02273  

The dataset is **non-residential only**: offices, retail, education, lodging, etc., from real buildings in North America and Europe.

---

## 2. Scale and coverage

| Aspect | Description |
|--------|-------------|
| **Total rows (train)** | ~53.6 million hourly meter readings |
| **Buildings** | 1,636 non-residential buildings |
| **Meters** | 3,053 energy meters (one or more per building) |
| **Sites** | 19 sites (geographic clusters; multiple buildings per site) |
| **Time span** | 2 full years: 2016 and 2017 |
| **Frequency** | Hourly (17,544 hours per meter over 2 years) |
| **Approx. size** | ~2–3 GB compressed (train + metadata + weather) |

Not every building has every meter type; the number of buildings per site varies (from single digits to hundreds).

---

## 2b. Region and North America focus

**Is the data specific to a region?**  
Yes. The data are from **19 sites in North America and Europe only** (no Asia, South America, etc.). Each site is a geographic cluster (city or area); all buildings in a site are within about 25 miles (40 km) of a central location.

**Can you focus the project only on North America?**  
Yes. The **Kaggle competition files** do not include a country or region column—only numeric `site_id` (0–18). To restrict to North America you have two options:

1. **Use timezone (recommended)**  
   The full **Building Data Genome Project 2 (BDG2)** metadata includes a **timezone** field per building (e.g. `US/Eastern`, `US/Pacific`, `Europe/London`). North America corresponds to timezones starting with `US/` (and possibly `America/` for Canada). You can:
   - Download the BDG2 metadata from [Zenodo](https://zenodo.org/records/3887306) or the [buds-lab/building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2) repo (or the [Kaggle BDG2 dataset](https://www.kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2)), which has `building_id`, `site_id` (animal names), and `timezone`.
   - Use the documented [BDG–Kaggle mapping](https://github.com/buds-lab/building-data-genome-project-2/wiki/BDG-Kaggle-mapping) to match Kaggle `building_id` to BDG2, then filter to rows where `timezone` is e.g. `US/Eastern`, `US/Central`, `US/Mountain`, or `US/Pacific` (and optionally Canada).
   - Keep only those `building_id`s (and their meters) in train, metadata, and weather when building your pipeline.

2. **Infer from Kaggle building_metadata**  
   If the competition’s `building_metadata.csv` includes a `timezone` column, filter directly to `timezone.str.startswith('US/')` (or your chosen North America timezones). If it does not, use the BDG2 metadata + mapping as above.

**Is there one specific region in North America?**  
No. North America in this dataset is **multiple regions**: the BDG2 documentation lists US timezones such as **US/Eastern**, **US/Central**, **US/Mountain**, and **US/Pacific**, so the sites span several US (and possibly Canadian) climates and locations. There is no single city or state that “is” the dataset—you can:
- Restrict to **all North America** (all US and optionally Canada timezones) for a continental focus, or  
- Restrict to **one timezone** (e.g. `US/Pacific` only) for a smaller, more homogeneous subset (fewer buildings, faster runs).

**Practical recommendation for “North America only”:**  
Use BDG2 metadata + BDG–Kaggle mapping to get a list of North America `building_id`s (or `site_id`s if you prefer to filter by site). Subset train, building_metadata, and weather to those buildings/sites before or during chunked processing. Document this filter in your report (e.g. “We restrict to North American sites using BDG2 timezone metadata”).

---

## 3. Files included

After downloading the competition data you typically get:

| File | Description |
|------|-------------|
| **train.csv** | Main training set: timestamp, building, meter type, and meter reading (target). |
| **test.csv** | Test set: same structure as train but without `meter_reading`; used for competition submission. |
| **building_metadata.csv** | One row per building: site, size, primary use, year built, etc. |
| **weather_train.csv** | Weather at each site for the training period (e.g. air temperature). |
| **weather_test.csv** | Weather at each site for the test period. |
| **sample_submission.csv** | Submission format: row_id and predicted meter_reading. |

Exact file names may vary slightly (e.g. `weather_train.csv` vs `weather_train.csv`); check the competition data page.

---

## 4. train.csv — main time series

This file is the core of the dataset: one row per (building, meter, timestamp).

| Column | Type | Description |
|--------|------|-------------|
| **building_id** | int | Unique building identifier. Links to `building_metadata.csv`. |
| **meter** | int | Meter type code (see below). |
| **timestamp** | datetime | Start of the hour (e.g. `2016-01-01 00:00:00`). |
| **meter_reading** | float | **Target variable.** Energy consumption for that hour. |

**Meter type codes (typical mapping):**

| Code | Meter type | Typical unit | Notes |
|------|------------|--------------|--------|
| 0 | Electricity | kWh | Whole-building electrical. **Site 0** uses **kBTU** (thousand BTU), not kWh. |
| 1 | Chilled water | kWh (or equivalent) | Cooling. |
| 2 | Steam | kWh (or equivalent) | Heating. |
| 3 | Hot water | kWh (or equivalent) | Heating. |

Some descriptions also mention **solar** and **water/irrigation** meters; if present, they will have additional meter codes in the data. Check the competition description or `train.csv` for the exact list.

**Important:**  
- **Missing or invalid readings:** Some rows may have missing or bad values (e.g. negative, zero, or extreme outliers). Cleaning and robustness (e.g. clipping, filtering) are part of the pipeline.  
- **Units:** Most readings are in kWh (or energy-equivalent). Site 0 electricity is in kBTU; you may want to convert or treat Site 0 separately for reporting.

---

## 5. building_metadata.csv — building attributes

One row per building. Used for features (e.g. size, use, age).

| Column | Type | Description |
|--------|------|-------------|
| **site_id** | int | Site (0–18). Groups buildings by location. |
| **building_id** | int | Same as in `train.csv`. |
| **primary_use** | str/category | Primary use of building (e.g. Education, Office, Retail, Lodging). |
| **square_feet** | float | Gross floor area (sq ft). |
| **year_built** | float | Year built; may have missing values. |

There may be additional columns depending on the exact competition release; check the data description on Kaggle.

**Usage:** Join to train on `building_id` to get site, primary_use, square_feet, year_built for feature engineering. Keep in memory (small table); join per chunk when processing train in chunks.

---

## 6. weather_train.csv and weather_test.csv

Weather at **site** level (not per building). One row per (site, timestamp).

| Column | Type | Description |
|--------|------|-------------|
| **site_id** | int | Site identifier. |
| **timestamp** | datetime | Start of the hour. |
| **air_temperature** | float | Air temperature (e.g. °C or °F; check competition). |
| **dew_temperature** | float | Dew point (if provided). |
| **cloud_coverage** | float | Cloud cover (if provided). |
| **precip_depth_1_hr** | float | Precipitation (if provided). |

Exact columns may vary; the main one for energy prediction is usually **air_temperature**.  
**Usage:** Load into memory (or by site). For each chunk of train, get `site_id` from building_metadata and join weather on `(site_id, timestamp)`.

---

## 7. test.csv and sample_submission.csv

- **test.csv:** Same columns as train **except** no `meter_reading`. Contains (building_id, meter, timestamp) for the competition test period.  
- **sample_submission.csv:** Typically `row_id` and `meter_reading` (to be filled with predictions).  

For a course project you can focus on train + building_metadata + weather_train; use a time-based split on train for validation (e.g. last 3–6 months) instead of the official test set if you prefer.

---

## 8. Data characteristics (for modeling)

- **Temporal:** Hourly, 2016–2017; strong daily and weekly seasonality; holidays and weather matter.  
- **Hierarchical:** Site → Building → Meter. Same site shares weather; same building can have multiple meter types.  
- **Heterogeneity:** Different primary_use, sizes, and meter types; not all buildings have all meters.  
- **Scale:** ~53.6M rows in train — use chunked or out-of-core processing; avoid loading the full file into memory at once.  
- **Target:** `meter_reading` is continuous (regression). Typical metrics: RMSE, CV-RMSE (coefficient of variation), or MAE.  

---

## 9. Typical pipeline (how the files connect)

1. **Load train in chunks** (e.g. 1–2M rows).  
2. **Join** each chunk to `building_metadata` on `building_id` → get site_id, primary_use, square_feet, year_built.  
3. **Join** to weather on `(site_id, timestamp)` → get air_temperature (and other weather if used).  
4. **Build leakage-safe features** (time, lags, rolling means, building, weather).  
5. **Train/validate** with a time-based split; predict `meter_reading`.  
6. **Anomaly:** residuals = actual − predicted; score per meter or per timestamp.  
7. **Decision support:** rank meters/buildings by anomaly or excess consumption for an audit list.

---

## 10. References

- **Competition:** https://www.kaggle.com/competitions/ashrae-energy-prediction  
- **Data description:** See the “Overview” and “Data” tabs on the competition page.  
- **Paper (Building Data Genome Project 2):** Miller et al. (2020), Scientific Data 7, 368; arXiv:2006.02273.  
- **Overview of competition and results:** arXiv:2007.06933 (ASHRAE GEPIII overview and results).

---

*This document summarizes the ASHRAE GEPIII dataset for the DS-GA 1019 project. Always confirm column names and units on the official Kaggle data page when you download.*
