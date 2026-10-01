"""
download_dataset.py
--------------------
Downloads the Deepfake Face Detection dataset directly from Kaggle using the
Kaggle API, extracts it automatically, and locates the dataset folder that
contains train/valid/test splits.

Kaggle authentication is picked up from either:
  1. A KAGGLE_API_TOKEN environment variable containing the full JSON
     contents of your kaggle.json (i.e. '{"username":"...","key":"..."}'), OR
  2. A kaggle.json file placed in the standard Kaggle config location
     (~/.kaggle/kaggle.json on Linux/Mac, C:\\Users\\<user>\\.kaggle\\kaggle.json
     on Windows).

No credentials are ever hard-coded in this file.
"""

import os
import sys
import json
import zipfile
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# TODO: Replace with the actual Kaggle dataset slug, e.g. "owner/dataset-name"
KAGGLE_DATASET = "PASTE_KAGGLE_DATASET_SLUG_HERE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
ZIP_PATH = DATASET_DIR / "dataset.zip"

REQUIRED_SPLITS = ["train", "valid", "test"]
# "validation" is accepted as an alias for "valid"
SPLIT_ALIASES = {"valid": ["valid", "validation", "val"]}
REQUIRED_CLASSES = ["real", "fake"]


def setup_kaggle_credentials():
    """
    Ensures Kaggle API credentials are available before the kaggle library
    is used. Supports either the KAGGLE_API_TOKEN env var (JSON string) or
    an existing kaggle.json in the default location.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_dir / "kaggle.json"

    if kaggle_json_path.exists():
        print("[Auth] Found existing kaggle.json credentials.")
        return True

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        try:
            creds = json.loads(token)
            assert "username" in creds and "key" in creds
        except (json.JSONDecodeError, AssertionError):
            print("[ERROR] KAGGLE_API_TOKEN is not valid JSON with "
                  "'username' and 'key' fields.")
            return False

        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json_path, "w") as f:
            json.dump(creds, f)
        # Kaggle API requires strict file permissions on Linux/Mac
        try:
            os.chmod(kaggle_json_path, 0o600)
        except OSError:
            pass
        print("[Auth] Wrote kaggle.json from KAGGLE_API_TOKEN env variable.")
        return True

    print("[ERROR] No Kaggle credentials found.")
    print("        Option 1: Place kaggle.json in ~/.kaggle/kaggle.json")
    print("        Option 2: Set the KAGGLE_API_TOKEN environment variable "
          "to the contents of kaggle.json")
    return False


def download_dataset():
    """Downloads the dataset zip from Kaggle if it isn't already present."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Skip download if a dataset folder already looks populated
    existing = [p for p in DATASET_DIR.iterdir() if p.is_dir()]
    if existing and not ZIP_PATH.exists():
        print(f"[Skip] Dataset directory already has content: {existing}")
        return True

    if ZIP_PATH.exists():
        print(f"[Skip] {ZIP_PATH.name} already downloaded.")
        return True

    if KAGGLE_DATASET == "PASTE_KAGGLE_DATASET_SLUG_HERE":
        print("[ERROR] KAGGLE_DATASET is not configured.")
        print("        Edit src/download_dataset.py and set KAGGLE_DATASET "
              "to the dataset slug from its Kaggle URL, e.g.:")
        print('        KAGGLE_DATASET = "owner-name/dataset-name"')
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] The 'kaggle' package is not installed. "
              "Run: pip install kaggle")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[ERROR] Kaggle authentication failed: {e}")
        return False

    try:
        print(f"[Download] Fetching dataset '{KAGGLE_DATASET}' from Kaggle...")
        api.dataset_download_files(
            KAGGLE_DATASET, path=str(DATASET_DIR), unzip=False, quiet=False
        )
    except Exception as e:
        print(f"[ERROR] Dataset download failed: {e}")
        return False

    # Kaggle API names the zip after the dataset; find and rename it
    downloaded_zips = list(DATASET_DIR.glob("*.zip"))
    if not downloaded_zips:
        print("[ERROR] Download completed but no ZIP file was found.")
        return False
    downloaded_zips[0].rename(ZIP_PATH)
    print(f"[Download] Saved to {ZIP_PATH}")
    return True


def extract_dataset():
    """Extracts the downloaded ZIP file into the dataset directory."""
    if not ZIP_PATH.exists():
        print("[Info] No ZIP file to extract (dataset may already be extracted).")
        return True

    try:
        print(f"[Extract] Extracting {ZIP_PATH.name}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATASET_DIR)
    except zipfile.BadZipFile as e:
        print(f"[ERROR] ZIP extraction failed - corrupted file: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] ZIP extraction failed: {e}")
        return False

    try:
        ZIP_PATH.unlink()
        print("[Extract] Removed ZIP file after extraction.")
    except OSError:
        pass

    return True


def _find_split_dir(root: Path, split_names):
    """Case-insensitive search for a directory matching any of split_names."""
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in split_names:
            return path
    return None


def find_dataset_root():
    """
    Walks the extracted dataset to find the directory that directly contains
    train/valid/test folders, regardless of how deeply nested the Kaggle
    ZIP structure is or what the top-level folder is named.
    """
    train_dir = _find_split_dir(DATASET_DIR, {"train"})
    if train_dir is None:
        return None
    return train_dir.parent


def verify_dataset_structure():
    """
    Verifies train/valid/test folders exist, each containing real/fake class
    folders, and prints image counts per class. Returns the resolved dataset
    root path on success, or None on failure.
    """
    dataset_root = find_dataset_root()
    if dataset_root is None:
        print("[ERROR] Could not locate a 'train' folder anywhere under "
              f"{DATASET_DIR}. The dataset structure does not match what "
              "was expected.")
        return None

    print(f"[Verify] Dataset root detected at: {dataset_root}")

    resolved_splits = {}
    for split in REQUIRED_SPLITS:
        aliases = SPLIT_ALIASES.get(split, [split])
        found = None
        for entry in dataset_root.iterdir():
            if entry.is_dir() and entry.name.lower() in aliases:
                found = entry
                break
        if found is None:
            print(f"[ERROR] Missing required split folder: '{split}' "
                  f"(also checked aliases {aliases}) under {dataset_root}")
            return None
        resolved_splits[split] = found

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    stats = {}
    for split, split_path in resolved_splits.items():
        stats[split] = {}
        for cls in REQUIRED_CLASSES:
            cls_dir = None
            for entry in split_path.iterdir():
                if entry.is_dir() and entry.name.lower() == cls:
                    cls_dir = entry
                    break
            if cls_dir is None:
                print(f"[ERROR] Missing class folder '{cls}' inside "
                      f"'{split}' split at {split_path}")
                return None

            count = sum(
                1 for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_exts
            )
            if count == 0:
                print(f"[ERROR] Class folder '{cls}' inside '{split}' is "
                      f"empty: {cls_dir}")
                return None
            stats[split][cls] = count

    print("\n[Dataset Statistics]")
    for split in REQUIRED_SPLITS:
        total = sum(stats[split].values())
        breakdown = ", ".join(f"{cls}={stats[split][cls]}" for cls in REQUIRED_CLASSES)
        print(f"  {split:5s}: total={total:6d}  ({breakdown})")

    return dataset_root


def run():
    print("[1/3] Checking Kaggle credentials...")
    if not setup_kaggle_credentials():
        sys.exit(1)

    print("[2/3] Downloading dataset (skipped if already present)...")
    if not download_dataset():
        sys.exit(1)

    if not extract_dataset():
        sys.exit(1)

    print("[3/3] Verifying dataset structure...")
    dataset_root = verify_dataset_structure()
    if dataset_root is None:
        sys.exit(1)

    print(f"\n[SUCCESS] Dataset ready at: {dataset_root}")
    return dataset_root


if __name__ == "__main__":
    run()
