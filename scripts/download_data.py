#!/usr/bin/env python3
"""Download ASHRAE Great Energy Predictor III dataset from Kaggle."""

from pathlib import Path
import subprocess
import zipfile
import sys

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMPETITION = "ashrae-energy-prediction"

EXPECTED_FILES = [
    "train.csv",
    "building_metadata.csv",
    "weather_train.csv",
    "weather_test.csv",
    "test.csv",
    "sample_submission.csv",
]


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{COMPETITION}.zip"

    if not zip_path.exists():
        print(f"Downloading {COMPETITION}...")
        try:
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
                check=True,
            )
        except FileNotFoundError:
            print("Kaggle CLI not found. Install: pip install kaggle")
            print("Then place your API key at ~/.kaggle/kaggle.json")
            return 1
        except subprocess.CalledProcessError:
            print("Download failed. Check credentials and competition rules acceptance.")
            return 1

    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print(f"Warning: missing files: {missing}")
        return 1

    print("Data ready. Files:")
    for f in sorted(DATA_DIR.iterdir()):
        if f.suffix == ".csv":
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {size_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(download())
