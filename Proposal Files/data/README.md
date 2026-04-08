# Project data

This folder is for the **ASHRAE Great Energy Predictor III** dataset.

## You don’t have the data yet

The dataset is **not** in this repo. You need to download it from Kaggle once.

## How to get the dataset

### Option 1: Kaggle API (recommended)

1. **Install the Kaggle CLI**
   ```bash
   pip install kaggle
   ```

2. **Get your API key**
   - Go to [Kaggle](https://www.kaggle.com) → Account → “Create New Token”.
   - This downloads `kaggle.json`. Put it at:
     - **Mac/Linux:** `~/.kaggle/kaggle.json`
     - **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`
   - Restrict permissions (Mac/Linux): `chmod 600 ~/.kaggle/kaggle.json`

3. **Download the competition data**
   - From the project root (parent of `data/`):
     ```bash
     kaggle competitions download -c ashrae-energy-prediction -p data/
     ```
   - Or run the script: `python scripts/download_data.py`

4. **Unzip**
   ```bash
   cd data && unzip ashrae-energy-prediction.zip && cd ..
   ```

### Option 2: Manual download

1. Open: https://www.kaggle.com/competitions/ashrae-energy-prediction/data
2. Sign in and click “Download All”.
3. Unzip the archive and place the CSV files inside this `data/` folder.

### Option 3: opendatasets (Python)

From the project root:

```bash
pip install opendatasets
python -c "import opendatasets; opendatasets.download('https://www.kaggle.com/competitions/ashrae-energy-prediction', data_dir='data')"
```

You’ll be prompted for your Kaggle username and API key (from Account → Create New Token).

---

## After download

Expected files (names may vary slightly):

- `train.csv` – main meter readings (~53M rows)
- `test.csv` – test period
- `building_metadata.csv` – building info
- `weather_train.csv` / `weather_test.csv` – weather
- `sample_submission.csv` – submission format

Total size is several GB. For a first run you can use a subset (e.g. one site or first N rows) to keep things fast.
