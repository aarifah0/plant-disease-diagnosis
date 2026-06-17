"""Download and extract the 38-class plant disease dataset from Kaggle.

Requires Kaggle API credentials: a kaggle.json token at ~/.kaggle/kaggle.json,
or the KAGGLE_USERNAME / KAGGLE_KEY environment variables. Get a token at
https://www.kaggle.com/settings -> API -> Create New Token.

Extraction is done manually (rather than relying on automatic unzip) because
this dataset's zip contains the same ~88k images twice -- once under an
upper-case top-level folder name and once under a lower-case duplicate -- and
the upper-case tree's nested path names exceed Windows' default 260-character
MAX_PATH limit when extracted naively.
"""
import argparse
import os
import zipfile
from pathlib import Path

DATASET_SLUG = "vipoooool/new-plant-diseases-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "_kaggle_download"
EXTRACT_ROOT = PROJECT_ROOT / "data" / "raw"

# Only extract this tree (the upper-case one); skip the byte-identical
# lower-case duplicate the zip also contains, and strip the doubled outer
# folder name so files land directly under <dest>/train and <dest>/valid.
INCLUDE_PREFIX = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/"


def _long_path(path_str: str) -> str:
    """Prefix an absolute Windows path with \\\\?\\ to bypass the 260-char MAX_PATH limit."""
    abs_path = os.path.abspath(path_str)
    return abs_path if abs_path.startswith("\\\\?\\") else f"\\\\?\\{abs_path}"


def extract_dataset_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract the train/ and valid/ trees from the dataset zip into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.startswith(INCLUDE_PREFIX) and not n.endswith("/")]
        total = len(members)
        print(f"Extracting {total} files to {dest_dir}...")
        for i, name in enumerate(members, 1):
            relative_parts = name[len(INCLUDE_PREFIX):].split("/")
            target = os.path.join(str(dest_dir), *relative_parts)
            os.makedirs(_long_path(os.path.dirname(target)), exist_ok=True)
            with zf.open(name) as src, open(_long_path(target), "wb") as dst:
                dst.write(src.read())
            if i % 5000 == 0 or i == total:
                print(f"  {i}/{total}")


def download_archive(dataset: str, download_dir: Path) -> Path:
    """Download the dataset zip (without auto-extracting) via the Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    download_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {dataset} via the Kaggle API to {download_dir}...")
    api.dataset_download_files(dataset, path=str(download_dir), unzip=False, quiet=False)

    zips = list(download_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip file found in {download_dir} after download.")
    return zips[0]


def main():
    parser = argparse.ArgumentParser(description="Download and extract the plant disease dataset from Kaggle.")
    parser.add_argument("--dataset", default=DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--archive", help="Path to an already-downloaded zip (skips downloading).")
    parser.add_argument("--dest", default=str(EXTRACT_ROOT), help="Directory to extract train/valid into.")
    args = parser.parse_args()

    zip_path = Path(args.archive) if args.archive else download_archive(args.dataset, DOWNLOAD_DIR)
    extract_dataset_zip(zip_path, Path(args.dest))
    print(f"\nDataset ready (pass this to src/train.py --data-dir): {args.dest}")


if __name__ == "__main__":
    main()
