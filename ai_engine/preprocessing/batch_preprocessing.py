
from pathlib import Path
from time import perf_counter
import json

import pandas as pd

from .text_cleaner import TextCleaner


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "data" / "processed" / "extracted"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "cleaned"
REPORT_DIR = BASE_DIR / "reports" / "preprocessing"

EXTRACTION_REPORT = (
    BASE_DIR / "reports" / "extraction" / "extraction_report.csv"
)


def load_extracted_documents():
    df = pd.read_csv(EXTRACTION_REPORT)

    df = df[df["status"].isin(["success", "skipped"])]

    return df.drop_duplicates(
        subset=["category", "file_name"]
    )


def preprocess_document(cleaner, category, file_name):
    start = perf_counter()

    file_stem = Path(file_name).stem

    input_path = INPUT_DIR / category / f"{file_stem}.txt"
    output_path = OUTPUT_DIR / category / f"{file_stem}.txt"

    record = {
        "file_name": file_name,
        "category": category,
        "status": None,
        "original_character_count": None,
        "cleaned_character_count": None,
        "original_word_count": None,
        "cleaned_word_count": None,
        "removed_characters": None,
        "reduction_percentage": None,
        "processing_time": None,
        "error": None
    }

    try:
        original_text = input_path.read_text(
            encoding="utf-8"
        )

        result = cleaner.clean(original_text)

        record.update({
            "original_character_count":
                result.original_character_count,
            "cleaned_character_count":
                result.cleaned_character_count,
            "original_word_count":
                result.original_word_count,
            "cleaned_word_count":
                result.cleaned_word_count,
            "removed_characters":
                result.removed_characters,
            "reduction_percentage":
                result.reduction_percentage
        })

        if result.is_empty:
            output_path.unlink(missing_ok=True)
            record["status"] = "excluded_empty"

        else:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path.write_text(
                result.cleaned_text,
                encoding="utf-8"
            )

            record["status"] = "success"

    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)

    record["processing_time"] = round(
        perf_counter() - start,
        4
    )

    return record


def generate_reports(records, execution_time):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    df.to_csv(
        REPORT_DIR / "preprocessing_report.csv",
        index=False
    )

    processed = df[
        df["status"].isin(["success", "excluded_empty"])
    ]

    original_characters = int(
        processed["original_character_count"].sum()
    )

    cleaned_characters = int(
        processed["cleaned_character_count"].sum()
    )

    summary = {
        "total_documents": len(df),
        "successful": int(
            df["status"].eq("success").sum()
        ),
        "excluded_empty": int(
            df["status"].eq("excluded_empty").sum()
        ),
        "failed": int(
            df["status"].eq("failed").sum()
        ),
        "original_characters": original_characters,
        "cleaned_characters": cleaned_characters,
        "removed_characters": (
            original_characters - cleaned_characters
        ),
        "character_reduction_percentage": round(
            (original_characters - cleaned_characters)
            / original_characters * 100
            if original_characters else 0.0,
            4
        ),
        "original_words": int(
            processed["original_word_count"].sum()
        ),
        "cleaned_words": int(
            processed["cleaned_word_count"].sum()
        ),
        "total_execution_time": round(execution_time, 2)
    }

    with (REPORT_DIR / "preprocessing_summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    category_summary = pd.crosstab(
        df["category"],
        df["status"]
    )

    category_summary.to_csv(
        REPORT_DIR / "category_summary.csv"
    )

    print("\nPREPROCESSING SUMMARY")
    print(json.dumps(summary, indent=4))

    print("\nCATEGORY SUMMARY")
    print(category_summary.to_string())

    return summary


def main():
    print("SMARTCV AI - Batch Text Preprocessing")

    documents = load_extracted_documents()

    if documents.empty:
        print("[ERROR] No extracted documents available")
        return 1

    cleaner = TextCleaner()
    records = []

    total = len(documents)
    start = perf_counter()

    print(f"Documents discovered: {total}")

    for index, row in enumerate(
        documents.itertuples(index=False),
        start=1
    ):
        record = preprocess_document(
            cleaner,
            row.category,
            row.file_name
        )

        records.append(record)

        print(
            f"[{index}/{total}] "
            f"{index / total * 100:.1f}% | "
            f"{record['status'].upper()} | "
            f"{row.category} | "
            f"{row.file_name}"
        )

    summary = generate_reports(
        records,
        perf_counter() - start
    )

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
