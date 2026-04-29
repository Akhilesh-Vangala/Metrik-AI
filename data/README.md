# Data

ASHRAE Great Energy Predictor III dataset. Three files needed:

| File | Size | Description |
|------|------|-------------|
| `train.csv` | 647 MB | Hourly meter readings — 20.2M rows |
| `building_metadata.csv` | 44 KB | Building attributes (site_id, primary_use, square_feet, year_built) |
| `weather_train.csv` | 14 MB | Hourly weather per site |

The other Kaggle files (`test.csv`, `weather_test.csv`, `sample_submission.csv`) are not used.

## Download

Accept the competition rules at kaggle.com/competitions/ashrae-energy-prediction/rules, then either:

**Kaggle CLI:** Set up `~/.kaggle/kaggle.json` and run `python3 scripts/download_data.py`

**Manual:** Download the three files from kaggle.com/competitions/ashrae-energy-prediction/data and place them in this directory.
