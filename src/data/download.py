"""Download the Kaggle eCommerce Events dataset (cosmetics shop).

Dataset: https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop
Contains ~20M events: view, cart, remove_from_cart, purchase with timestamps,
session IDs, product info, and user IDs.

Prerequisites:
    pip install kaggle
    Set up ~/.kaggle/kaggle.json with your API credentials.
"""

import os
import subprocess
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(RAW_DIR.glob("*.csv"))
    if csv_files:
        print(f"Data already exists in {RAW_DIR} ({len(csv_files)} CSV files). Skipping download.")
        return csv_files

    print("Downloading eCommerce Events dataset from Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "mkechinov/ecommerce-events-history-in-cosmetics-shop",
            "-p", str(RAW_DIR),
        ],
        check=True,
    )

    for zip_file in RAW_DIR.glob("*.zip"):
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_file.unlink()

    csv_files = list(RAW_DIR.glob("*.csv"))
    print(f"Downloaded {len(csv_files)} CSV files to {RAW_DIR}")
    return csv_files


if __name__ == "__main__":
    download_dataset()
