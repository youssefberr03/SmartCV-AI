
from pathlib import Path
from collections import Counter
from time import perf_counter
import argparse
import json

import pandas as pd

from .linguistic_analyzer import LinguisticAnalyzer


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "data" / "processed" / "cleaned"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "linguistic"
REPORT_DIR = BASE_DIR / "reports" / "linguistic"


def discover_documents():
    return sorted(INPUT_DIR.rglob("*.txt"))


def prepare_documents(paths, analyzer, records):
    groups = {
        "en": [],
        "fr": []
    }

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            language = analyzer.detect_language(text)

            context = {
                "file_name": path.name,
                "category": path.parent.name,
                "language": language,
                "source_path": str(
                    path.relative_to(INPUT_DIR)
                )
            }

            if language not in groups:
                records.append({
                    **context,
                    "status": "unsupported_language"
                })

                continue

            groups[language].append((text, context))

        except Exception as error:
            records.append({
                "file_name": path.name,
                "category": path.parent.name,
                "status": "failed",
                "error": str(error)
            })

    return groups


def save_result(result, context):
    category = context["category"]
    stem = Path(context["file_name"]).stem

    output_dir = OUTPUT_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)

    text_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    text_path.write_text(
        result.lemmatized_text,
        encoding="utf-8"
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            result.to_dict(),
            file,
            ensure_ascii=False
        )


def process_group(
    analyzer,
    documents,
    language,
    records,
    global_pos,
    global_entities
):
    total = len(documents)

    print(f"\nProcessing language: {language}")
    print(f"Documents: {total}")

    try:
        analyzer.load_model(language)

    except Exception as error:
        for _, context in documents:
            records.append({
                **context,
                "status": "failed",
                "error": str(error)
            })

        return

    completed = set()

    try:
        for index, (result, context) in enumerate(
            analyzer.analyze_batch(
                documents,
                language,
                batch_size=32
            ),
            start=1
        ):
            record = dict(context)

            try:
                save_result(result, context)

                record.update({
                    "status": "success",
                    "token_count": result.token_count,
                    "unique_token_count":
                        result.unique_token_count,
                    "stop_word_count":
                        result.stop_word_count,
                    "punctuation_count":
                        result.punctuation_count,
                    "sentence_count":
                        result.sentence_count,
                    "vocabulary_size":
                        result.vocabulary_size,
                    "entity_count": len(result.entities)
                })

                global_pos.update(
                    result.pos_distribution
                )

                global_entities.update(
                    entity.label
                    for entity in result.entities
                )

            except Exception as error:
                record.update({
                    "status": "failed",
                    "error": str(error)
                })

            records.append(record)
            completed.add(context["source_path"])

            print(
                f"[{index}/{total}] "
                f"{record['status'].upper()} | "
                f"{context['file_name']}"
            )

    except Exception as error:
        for _, context in documents:
            if context["source_path"] not in completed:
                records.append({
                    **context,
                    "status": "failed",
                    "error": str(error)
                })


def generate_reports(
    records,
    global_pos,
    global_entities,
    execution_time
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    df.to_csv(
        REPORT_DIR / "linguistic_report.csv",
        index=False
    )

    successful = df[df["status"].eq("success")]

    summary = {
        "total_documents": len(df),
        "successful": len(successful),
        "failed": int(
            df["status"].eq("failed").sum()
        ),
        "unsupported_language": int(
            df["status"].eq(
                "unsupported_language"
            ).sum()
        ),
        "total_tokens": 0,
        "total_stop_words": 0,
        "total_punctuation": 0,
        "total_sentences": 0,
        "total_entities": 0,
        "execution_time": round(execution_time, 2)
    }

    metrics = {
        "total_tokens": "token_count",
        "total_stop_words": "stop_word_count",
        "total_punctuation": "punctuation_count",
        "total_sentences": "sentence_count",
        "total_entities": "entity_count"
    }

    if not successful.empty:
        for key, column in metrics.items():
            summary[key] = int(
                successful[column].sum()
            )

        successful["language"].value_counts().to_csv(
            REPORT_DIR / "language_distribution.csv"
        )

    pd.Series(
        dict(global_pos.most_common()),
        dtype="int64"
    ).to_csv(
        REPORT_DIR / "pos_distribution.csv",
        header=["count"]
    )

    pd.Series(
        dict(global_entities.most_common()),
        dtype="int64"
    ).to_csv(
        REPORT_DIR / "entity_distribution.csv",
        header=["count"]
    )

    with (
        REPORT_DIR / "linguistic_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nLINGUISTIC ANALYSIS SUMMARY")
    print(json.dumps(summary, indent=4))

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

    paths = discover_documents()

    if args.limit is not None:
        paths = paths[:args.limit]

    if not paths:
        print("[ERROR] No cleaned documents found")
        return 1

    print("SMARTCV AI - Linguistic Analysis")
    print(f"Documents discovered: {len(paths)}")

    start = perf_counter()

    analyzer = LinguisticAnalyzer()
    records = []

    global_pos = Counter()
    global_entities = Counter()

    groups = prepare_documents(
        paths,
        analyzer,
        records
    )

    for language, documents in groups.items():
        if documents:
            process_group(
                analyzer,
                documents,
                language,
                records,
                global_pos,
                global_entities
            )

    summary = generate_reports(
        records,
        global_pos,
        global_entities,
        perf_counter() - start
    )

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
