# Data

This directory holds the ASHRAE Great Energy Predictor III dataset.

**Files are NOT committed to the repo** (too large). Download them before running the pipeline.

## Download

```bash
python scripts/download_data.py
```

Or manually from [Kaggle](https://www.kaggle.com/competitions/ashrae-energy-prediction/data).

## Expected Files

| File | Rows | Size | Description |
|------|------|------|-------------|
| `train.csv` | 20,216,100 | 647 MB | Hourly meter readings (building_id, meter, timestamp, meter_reading) |
| `building_metadata.csv` | 1,449 | 44 KB | Building attributes (site_id, primary_use, square_feet, year_built) |
| `weather_train.csv` | 139,773 | 7.1 MB | Hourly weather per site (air_temperature, dew_temperature, etc.) |
| `test.csv` | 41,697,600 | 1.4 GB | Competition test set (no meter_reading — not used for training) |
| `weather_test.csv` | ~277K | 14 MB | Weather for test period |
| `sample_submission.csv` | 41,697,600 | 427 MB | Competition submission template |

## Kaggle API Setup

1. Create an account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → Create New Token (downloads `kaggle.json`)
3. Move it: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
5. Accept competition rules at the [competition page](https://www.kaggle.com/competitions/ashrae-energy-prediction/rules)
