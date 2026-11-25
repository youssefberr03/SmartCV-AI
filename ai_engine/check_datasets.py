
from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    "Resume Dataset": "resumes/Resume.csv",
    "Job Postings": "jobs/postings.csv",
    "ESCO English": "esco/en/skills_en.csv",
    "ESCO French": "esco/fr/skills_fr.csv",
    "Matching Resumes": "matching/resumes.jsonl",
    "Matching Jobs": "matching/jobs.jsonl",
    "Matching References": "matching/matches.jsonl"
}


def load_dataset(path):
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported format: {path.suffix}")


def check_dataset(name, relative_path):
    path = RAW_DIR / relative_path

    print(f"\n===== {name.upper()} =====")

    if not path.is_file():
        print(f"[MISSING] {path}")
        return False

    try:
        df = load_dataset(path)

        size_mb = path.stat().st_size / (1024 ** 2)

        print(f"File: {path.name}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        print(f"Column names: {df.columns.tolist()}")

        print("\nMissing values:")
        print(df.isna().sum().to_string())

        if df.empty:
            print("[ERROR] Empty dataset")
            return False

        print("\n[OK] Dataset loaded successfully")
        return True

    except Exception as error:
        print(f"[ERROR] {error}")
        return False


def main():
    print("SMARTCV AI - Dataset Verification")

    results = {}

    for name, relative_path in DATASETS.items():
        results[name] = check_dataset(name, relative_path)

    print("\n===== DATASET SUMMARY =====")

    for name, status in results.items():
        label = "OK" if status else "FAILED"
        print(f"[{label}] {name}")

    success_count = sum(results.values())

    print(f"\nAvailable datasets: {success_count}/{len(DATASETS)}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
