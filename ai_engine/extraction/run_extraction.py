
from pathlib import Path
import argparse
import json

from .pdf_extractor import PDFExtractor


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "extracted"


def save_results(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_stem = Path(result.file_name).stem

    text_path = OUTPUT_DIR / f"{file_stem}.txt"
    json_path = OUTPUT_DIR / f"{file_stem}.json"

    text_path.write_text(
        result.extracted_text,
        encoding="utf-8"
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            result.to_dict(),
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nText saved to: {text_path}")
    print(f"JSON saved to: {json_path}")


def display_results(result):
    print("\nPDF EXTRACTION RESULT")

    print(f"File: {result.file_name}")
    print(f"Pages: {result.page_count}")
    print(f"Size: {result.file_size_bytes} bytes")
    print(f"Characters: {result.character_count}")
    print(f"Words: {result.word_count}")
    print(f"OCR required: {result.requires_ocr}")

    for page in result.pages:
        print(f"\nPage {page.page_number}")
        print(f"Characters: {page.character_count}")
        print(f"Words: {page.word_count}")
        print(f"Images: {page.image_count}")
        print(f"Text blocks: {len(page.blocks)}")

    if result.warnings:
        print("\nWarnings:")

        for warning in result.warnings:
            print(f"- {warning}")


def main():
    parser = argparse.ArgumentParser(
        description="SmartCV AI PDF Extraction"
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the PDF file"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display a preview of extracted text"
    )

    args = parser.parse_args()

    extractor = PDFExtractor()

    try:
        result = extractor.extract(args.file)

        display_results(result)
        save_results(result)

        if args.preview:
            print("\nEXTRACTED TEXT PREVIEW")
            print(result.extracted_text[:1500])

        print("\n[OK] PDF extraction completed")

        return 0

    except Exception as error:
        print(f"\n[ERROR] {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
