#!/usr/bin/env python3
"""
Download ASHRAE Great Energy Predictor III dataset from Kaggle.
Run from project root. Requires: pip install kaggle, and ~/.kaggle/kaggle.json set up.
"""
from pathlib import Path
import subprocess
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COMPETITION = "ashrae-energy-prediction"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{COMPETITION}.zip"

    if zip_path.exists():
        print(f"Found {zip_path}. Extracting...")
    else:
        print(f"Downloading {COMPETITION} into {DATA_DIR}...")
        try:
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
                check=True,
            )
        except FileNotFoundError:
            print("Kaggle CLI not found. Install with: pip install kaggle")
            print("Then put your API key at ~/.kaggle/kaggle.json")
            return 1
        except subprocess.CalledProcessError as e:
            print("Download failed. Check your Kaggle credentials and that you've accepted the competition rules.")
            return e.returncode

    if zip_path.exists():
        print("Unzipping...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print("Done. Files in:", DATA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
