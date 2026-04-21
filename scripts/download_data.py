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
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except FileNotFoundError:
            print("\nERROR: Kaggle CLI not found.")
            print("1. Install it via: pip install kaggle")
            print("2. Place your API key at ~/.kaggle/kaggle.json\n")
            return 1
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.lower() if e.stderr else ""
            if "403" in err_msg or "forbidden" in err_msg:
                print("\n" + "="*65)
                print(" ERROR: 403 Forbidden - Kaggle Rules Not Accepted")
                print("="*65)
                print("Kaggle requires you to accept the competition rules to download.")
                print("Please follow these quick steps:")
                print(f"  1. Go to: https://www.kaggle.com/competitions/{COMPETITION}/rules")
                print("  2. Log in and click 'I Understand and Accept'")
                print("  3. Re-run this script")
                print("="*65 + "\n")
            elif "401" in err_msg or "unauthorized" in err_msg:
                print("\n" + "="*65)
                print(" ERROR: 401 Unauthorized - Invalid Kaggle API Key")
                print("="*65)
                print("Your ~/.kaggle/kaggle.json file is missing or invalid.")
                print("Go to kaggle.com -> Account -> Create New API Token,")
                print("and save the downloaded file to ~/.kaggle/kaggle.json")
                print("="*65 + "\n")
            else:
                print(f"\nDownload failed with error:\n{e.stderr or e.stdout}\n")
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
