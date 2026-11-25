
from pathlib import Path
import json

import pandas as pd

from check_datasets import DATASETS, RAW_DIR, load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "reports" / "datasets"


def create_inventory():
    inventory = []

    for name, relative_path in DATASETS.items():
        path = RAW_DIR / relative_path

        entry = {
            "dataset": name,
            "path": relative_path,
            "available": path.is_file()
        }

        if path.is_file():
            try:
                df = load_dataset(path)

                entry.update({
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "size_mb": round(
                        path.stat().st_size / (1024 ** 2),
                        3
                    ),
                    "format": path.suffix,
                    "column_names": df.columns.tolist()
                })

            except Exception as error:
                entry["error"] = str(error)

        inventory.append(entry)

    return inventory


def save_inventory(inventory):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "dataset_inventory.json"
    csv_path = OUTPUT_DIR / "dataset_inventory.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            inventory,
            file,
            indent=4,
            ensure_ascii=False
        )

    pd.DataFrame(inventory).to_csv(
        csv_path,
        index=False
    )

    print(f"JSON inventory: {json_path}")
    print(f"CSV inventory: {csv_path}")


def main():
    inventory = create_inventory()
    save_inventory(inventory)

    df = pd.DataFrame(inventory)

    columns = [
        column
        for column in [
            "dataset",
            "available",
            "rows",
            "columns",
            "size_mb"
        ]
        if column in df.columns
    ]

    print("\n===== DATASET INVENTORY =====")
    print(df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
