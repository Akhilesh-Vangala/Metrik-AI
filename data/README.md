# Data

This directory holds the ASHRAE Great Energy Predictor III dataset.

**Files are NOT committed to the repo** (too large for git). Download them before running the pipeline.

The pipeline only needs three files:

| File | Size | Description |
|------|------|-------------|
| `train.csv` | 647 MB | Hourly meter readings — 20.2M rows (building_id, meter, timestamp, meter_reading) |
| `building_metadata.csv` | 44 KB | Building attributes (site_id, primary_use, square_feet, year_built) |
| `weather_train.csv` | 14 MB | Hourly weather per site (air_temperature, dew_temperature, wind_speed, etc.) |

The other Kaggle files (`test.csv`, `weather_test.csv`, `sample_submission.csv`) are not used and do not need to be downloaded.

---

## Option A — Kaggle CLI (recommended)

**Step 1 — Install the Kaggle CLI:**
```bash
pip install kaggle
```

**Step 2 — Set up your API key:**
1. Create an account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → Create New API Token (downloads `kaggle.json`)
3. Move it: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Step 3 — Accept competition rules:**

Go to [kaggle.com/competitions/ashrae-energy-prediction/rules](https://www.kaggle.com/competitions/ashrae-energy-prediction/rules), log in, and click **I Understand and Accept**. The download will fail with a 403 error if this step is skipped.

**Step 4 — Run the download script:**
```bash
python3 scripts/download_data.py
```

The script downloads, extracts, and verifies all files automatically.

---

## Option B — Manual download

1. Go to [kaggle.com/competitions/ashrae-energy-prediction/data](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)
2. Log in and accept the competition rules if prompted
3. Download `train.csv`, `building_metadata.csv`, and `weather_train.csv`
4. Place all three files in this `data/` directory
