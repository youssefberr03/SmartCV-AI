
from pathlib import Path
from time import perf_counter
import argparse
import json

import pandas as pd

from .pdf_extractor import PDFExtractor


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "data" / "raw" / "resumes"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "extracted"
REPORT_DIR = BASE_DIR / "reports" / "extraction"


def discover_pdfs():
    return sorted(INPUT_DIR.rglob("*.pdf"))


def save_extraction(result, text_path, json_path):
    text_path.write_text(
        result.extracted_text,
        encoding="utf-8"
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            result.to_dict(),
            file,
            indent=4,
            ensure_ascii=False
        )


def extract_document(extractor, pdf_path, overwrite=False):
    start = perf_counter()

    category = pdf_path.parent.name
    category_dir = OUTPUT_DIR / category

    category_dir.mkdir(parents=True, exist_ok=True)

    text_path = category_dir / f"{pdf_path.stem}.txt"
    json_path = category_dir / f"{pdf_path.stem}.json"

    record = {
        "file_name": pdf_path.name,
        "category": category,
        "source_path": str(pdf_path.relative_to(INPUT_DIR)),
        "status": None,
        "page_count": None,
        "character_count": None,
        "word_count": None,
        "requires_ocr": None,
        "processing_time": None,
        "error": None
    }

    try:
        if (
            not overwrite
            and text_path.is_file()
            and json_path.is_file()
        ):
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            record.update({
                "status": "skipped",
                "page_count": data["page_count"],
                "character_count": data["character_count"],
                "word_count": data["word_count"],
                "requires_ocr": data["requires_ocr"]
            })

        else:
            result = extractor.extract(pdf_path)

            save_extraction(
                result,
                text_path,
                json_path
            )

            record.update({
                "status": "success",
                "page_count": result.page_count,
                "character_count": result.character_count,
                "word_count": result.word_count,
                "requires_ocr": result.requires_ocr
            })

    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)

    record["processing_time"] = round(
        perf_counter() - start,
        4
    )

    return record


def generate_reports(records, total_time):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    df.to_csv(
        REPORT_DIR / "extraction_report.csv",
        index=False
    )

    available = df[
        df["status"].isin(["success", "skipped"])
    ]

    summary = {
        "total_documents": len(df),
        "successful": int((df["status"] == "success").sum()),
        "skipped": int((df["status"] == "skipped").sum()),
        "failed": int((df["status"] == "failed").sum()),
        "available_documents": len(available),
        "empty_documents": int(
            available["character_count"].eq(0).sum()
        ),
        "ocr_required": int(
            available["requires_ocr"].eq(True).sum()
        ),
        "total_pages": int(
            available["page_count"].sum()
        ),
        "total_words": int(
            available["word_count"].sum()
        ),
        "total_characters": int(
            available["character_count"].sum()
        ),
        "total_execution_time": round(total_time, 2)
    }

    with (REPORT_DIR / "extraction_summary.json").open(
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

    print("\nBATCH EXTRACTION SUMMARY")
    print(json.dumps(summary, indent=4))

    print("\nCATEGORY SUMMARY")
    print(category_summary.to_string())

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="SmartCV AI Batch PDF Extraction"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    pdf_files = discover_pdfs()

    print(f"PDF files discovered: {len(pdf_files)}")

    if args.limit is not None:
        pdf_files = pdf_files[:args.limit]

    if not pdf_files:
        print("[ERROR] No PDF files found")
        return 1

    extractor = PDFExtractor()
    records = []

    total = len(pdf_files)
    start = perf_counter()

    for index, pdf_path in enumerate(pdf_files, start=1):
        record = extract_document(
            extractor,
            pdf_path,
            overwrite=args.overwrite
        )

        records.append(record)

        percentage = index / total * 100

        print(
            f"[{index}/{total}] "
            f"{percentage:.1f}% | "
            f"{record['status'].upper()} | "
            f"{record['category']} | "
            f"{pdf_path.name}"
        )

    summary = generate_reports(
        records,
        perf_counter() - start
    )

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
