
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data/raw/matching"

DATASET_NAME = "michaelozon/candidate-matching-synthetic"
DATASET_FOLDERS = ["resumes", "jobs", "matches"]


def download_dataset(name):
    print(f"\nDownloading {name}...")

    file_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=f"{name}/train-00000-of-00001.parquet",
        repo_type="dataset"
    )

    df = pd.read_parquet(file_path)

    output_path = OUTPUT_DIR / f"{name}.jsonl"

    df.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"Records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Saved to: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in DATASET_FOLDERS:
        download_dataset(name)

    print("\nAll matching datasets downloaded successfully!")


if __name__ == "__main__":
    main()
