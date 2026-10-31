"""Download the FIFA 23 Complete Player Dataset from Kaggle into data/raw/.

kagglehub can fetch public datasets without a Kaggle login.
"""

from pathlib import Path

import kagglehub

DATASET = "stefanoleone992/fifa-23-complete-player-dataset"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(DATASET, output_dir=str(RAW_DIR))
    print(f"Downloaded to {path}")
    for f in sorted(RAW_DIR.iterdir()):
        print(f"  {f.name:30s} {f.stat().st_size / 1e9:6.2f} GB")


if __name__ == "__main__":
    main()
