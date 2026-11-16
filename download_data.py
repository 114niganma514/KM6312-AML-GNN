# download the elliptic dataset from kaggle

import os
import sys
import zipfile
import shutil

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = BASE_DIR
KAGGLE_DATASET = "ellipticco/elliptic-data-set"


def check_kaggle_credentials():
    """check if Kaggle API credentials are available"""
    # check for kaggle.json
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    # check for environment variables
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if os.path.exists(kaggle_json) or has_env:
        return True
    return False


def download_dataset():
    """download the Elliptic dataset from Kaggle"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package is not installed.")
        print("Install with: pip install kaggle")
        sys.exit(1)

    if not check_kaggle_credentials():
        print("=" * 60)
        print("ERROR: Kaggle API credentials not found!")
        print("=" * 60)
        print("\nPlease set up Kaggle API credentials:")
        print("  1. Go to https://www.kaggle.com/settings/api")
        print("  2. Click 'Create New Token' to download kaggle.json")
        print(f"  3. Save it to: {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}")
        print("\nOr set environment variables:")
        print("  set KAGGLE_USERNAME=your_username")
        print("  set KAGGLE_KEY=your_api_key")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print(f"Downloading Elliptic Dataset from Kaggle: {KAGGLE_DATASET}")
    print("=" * 60)

    # initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # create output directory
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # download dataset zip
    zip_path = os.path.join(DATA_RAW_DIR, "elliptic-data-set.zip")
    print(f"\nDownloading to: {zip_path}")
    api.dataset_download_files(KAGGLE_DATASET, path=DATA_RAW_DIR, unzip=False)

    # extract zip
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_RAW_DIR)

    # remove zip
    os.remove(zip_path)
    print("Zip file removed.")

    # verify files
    required_files = [
        "elliptic_txs_features.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_classes.csv",
    ]

    print("\nVerifying downloaded files:")
    all_present = True
    for f in required_files:
        fpath = os.path.join(DATA_RAW_DIR, f)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  [OK] {f} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {f}")
            all_present = False

    if all_present:
        print("\n" + "=" * 60)
        print("SUCCESS: Dataset downloaded and verified!")
        print("=" * 60)
        print(f"\nData directory: {DATA_RAW_DIR}")
        print("\nYou can now run the full pipeline with:")
        print("  python run_all.py")
    else:
        print("\nWARNING: Some files are missing. Please check the download.")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
