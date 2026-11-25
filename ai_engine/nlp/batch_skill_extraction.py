
from pathlib import Path
from collections import Counter
from time import perf_counter
import argparse
import json

import pandas as pd

from .skill_extractor import SkillExtractor


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "data" / "processed" / "cleaned"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "skills"
REPORT_DIR = BASE_DIR / "reports" / "skills"

LINGUISTIC_REPORT = (
    BASE_DIR
    / "reports"
    / "linguistic"
    / "linguistic_report.csv"
)


def load_documents():
    df = pd.read_csv(LINGUISTIC_REPORT)

    return df[
        df["status"].eq("success")
        & df["language"].isin(["en", "fr"])
    ].drop_duplicates(
        subset=["source_path"]
    )


def process_document(extractor, row):
    start = perf_counter()

    source_path = Path(row.source_path)
    input_path = INPUT_DIR / source_path

    output_path = (
        OUTPUT_DIR / source_path
    ).with_suffix(".json")

    record = {
        "file_name": row.file_name,
        "category": row.category,
        "language": row.language,
        "status": None,
        "skill_count": None,
        "negated_skill_count": None,
        "processing_time": None,
        "error": None
    }

    skills = []

    try:
        text = input_path.read_text(
            encoding="utf-8"
        )

        result = extractor.extract(
            text,
            language=row.language
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2
            )

        skills = result["skills"]

        record.update({
            "status": "success",
            "skill_count": result["skill_count"],
            "negated_skill_count":
                result["negated_skill_count"]
        })

    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)

    record["processing_time"] = round(
        perf_counter() - start,
        4
    )

    return record, skills


def generate_reports(
    records,
    skill_frequency,
    skill_labels,
    category_frequency,
    execution_time
):
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(records)

    df.to_csv(
        REPORT_DIR / "skill_extraction_report.csv",
        index=False
    )

    successful = df[
        df["status"].eq("success")
    ]

    summary = {
        "total_documents": len(df),
        "successful": len(successful),
        "failed": int(
            df["status"].eq("failed").sum()
        ),
        "documents_without_skills": int(
            successful["skill_count"].eq(0).sum()
        ),
        "total_document_skill_pairs": int(
            successful["skill_count"].sum()
        ),
        "unique_skills_detected": len(skill_frequency),
        "negated_skills": int(
            successful["negated_skill_count"].sum()
        ),
        "average_skills_per_document": round(
            float(successful["skill_count"].mean()),
            2
        ) if not successful.empty else 0,
        "total_execution_time": round(
            execution_time,
            2
        )
    }

    frequency_records = [
        {
            "skill_id": skill_id,
            "skill": skill_labels[skill_id],
            "document_frequency": frequency
        }
        for skill_id, frequency
        in skill_frequency.most_common()
    ]

    frequency_df = pd.DataFrame(
        frequency_records,
        columns=[
            "skill_id",
            "skill",
            "document_frequency"
        ]
    )

    frequency_df.to_csv(
        REPORT_DIR / "skill_frequency.csv",
        index=False
    )

    category_records = [
        {
            "category": category,
            "skill_id": skill_id,
            "skill": skill_labels[skill_id],
            "document_frequency": frequency
        }
        for (category, skill_id), frequency
        in category_frequency.most_common()
    ]

    pd.DataFrame(
        category_records,
        columns=[
            "category",
            "skill_id",
            "skill",
            "document_frequency"
        ]
    ).to_csv(
        REPORT_DIR / "skills_by_category.csv",
        index=False
    )

    with (
        REPORT_DIR / "skill_extraction_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nSKILL EXTRACTION SUMMARY")
    print(json.dumps(summary, indent=4))

    print("\nTOP 20 SKILLS")
    print(
        frequency_df[
            ["skill", "document_frequency"]
        ].head(20).to_string(index=False)
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    documents = load_documents()

    if args.limit is not None:
        documents = documents.head(args.limit)

    if documents.empty:
        print("[ERROR] No documents available")
        return 1

    extractor = SkillExtractor()

    records = []
    skill_frequency = Counter()
    category_frequency = Counter()
    skill_labels = {}

    total = len(documents)
    start = perf_counter()

    print("SMARTCV AI - Batch Skill Extraction")
    print(f"Documents: {total}")

    for index, row in enumerate(
        documents.itertuples(index=False),
        start=1
    ):
        record, skills = process_document(
            extractor,
            row
        )

        records.append(record)

        for skill in skills:
            skill_id = skill["id"]

            skill_labels[skill_id] = skill["label"]

            skill_frequency[skill_id] += 1

            category_frequency[
                (row.category, skill_id)
            ] += 1

        print(
            f"[{index}/{total}] "
            f"{record['status'].upper()} | "
            f"{row.file_name} | "
            f"Skills: {record['skill_count']}"
        )

    summary = generate_reports(
        records,
        skill_frequency,
        skill_labels,
        category_frequency,
        perf_counter() - start
    )

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
