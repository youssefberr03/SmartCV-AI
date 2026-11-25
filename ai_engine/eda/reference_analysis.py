
from pathlib import Path
from collections import Counter
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "eda"


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def analyze_esco():
    print("\nAnalyzing ESCO...")

    output_dir = REPORT_DIR / "esco"
    output_dir.mkdir(parents=True, exist_ok=True)

    english = pd.read_csv(
        RAW_DIR / "esco" / "en" / "skills_en.csv"
    )

    french = pd.read_csv(
        RAW_DIR / "esco" / "fr" / "skills_fr.csv"
    )

    english_ids = set(english["conceptUri"].dropna())
    french_ids = set(french["conceptUri"].dropna())

    common_ids = english_ids & french_ids

    statistics = {
        "english_rows": len(english),
        "french_rows": len(french),
        "english_unique_ids": len(english_ids),
        "french_unique_ids": len(french_ids),
        "common_ids": len(common_ids),
        "english_only_ids": len(english_ids - french_ids),
        "french_only_ids": len(french_ids - english_ids),
        "english_duplicate_ids": int(
            english["conceptUri"].dropna().duplicated().sum()
        ),
        "french_duplicate_ids": int(
            french["conceptUri"].dropna().duplicated().sum()
        ),
        "english_missing_ids": int(
            english["conceptUri"].isna().sum()
        ),
        "french_missing_ids": int(
            french["conceptUri"].isna().sum()
        )
    }

    for language, df in [("en", english), ("fr", french)]:
        df["skillType"].value_counts().to_csv(
            output_dir / f"skill_types_{language}.csv"
        )

        df["reuseLevel"].value_counts().to_csv(
            output_dir / f"reuse_levels_{language}.csv"
        )

        df["status"].value_counts().to_csv(
            output_dir / f"status_{language}.csv"
        )

        df.isna().sum().to_csv(
            output_dir / f"missing_values_{language}.csv"
        )

    save_json(
        statistics,
        output_dir / "esco_statistics.json"
    )

    print(json.dumps(statistics, indent=4))


def analyze_matching():
    print("\nAnalyzing Candidate Matching Dataset...")

    output_dir = REPORT_DIR / "matching"
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics = {}

    for name in ["resumes", "jobs", "matches"]:
        path = RAW_DIR / "matching" / f"{name}.jsonl"

        df = pd.read_json(path, lines=True)

        dataset_info = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "missing_values": {
                column: int(value)
                for column, value in df.isna().sum().items()
            },
            "column_structures": {}
        }

        for column in df.columns:
            values = df[column].dropna()

            types = Counter(
                type(value).__name__
                for value in values
            )

            column_info = {
                "value_types": dict(types)
            }

            list_lengths = [
                len(value)
                for value in values
                if isinstance(value, list)
            ]

            if list_lengths:
                lengths = pd.Series(list_lengths)

                column_info["list_statistics"] = {
                    "list_count": len(lengths),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "average_length": float(lengths.mean()),
                    "total_elements": int(lengths.sum())
                }

            dataset_info["column_structures"][column] = (
                column_info
            )

        statistics[name] = dataset_info

        print(f"\n{name.upper()}")
        print(json.dumps(dataset_info, indent=4))

    save_json(
        statistics,
        output_dir / "matching_statistics.json"
    )
