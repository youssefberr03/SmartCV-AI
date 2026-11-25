
from pathlib import Path
from collections import defaultdict
from time import perf_counter
import argparse
import json
import random

import pandas as pd

from ai_engine.nlp.skill_repository import SkillRepository
from ai_engine.nlp.skill_extractor import SkillExtractor
from ai_engine.preprocessing.text_cleaner import TextCleaner

from .skill_metrics import (
    evaluate_prediction,
    summarize_metrics
)


BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = BASE_DIR / "data" / "evaluation"

DEFAULT_DATASET = EVALUATION_DIR / "skill_gold.json"

REAL_DATASET = (
    EVALUATION_DIR / "real_cv_annotations.json"
)

REPORT_DIR = BASE_DIR / "reports" / "evaluation"

CLEANED_DIR = (
    BASE_DIR / "data" / "processed" / "cleaned"
)

LINGUISTIC_REPORT = (
    BASE_DIR
    / "reports"
    / "linguistic"
    / "linguistic_report.csv"
)

VALID_STATUSES = {
    "asserted",
    "mentioned",
    "negated",
    "learning",
    "interest",
    "conflicting"
}


class SkillEvaluator:

    def __init__(self):
        self.repository = SkillRepository().load()

        self.extractor = SkillExtractor(
            self.repository
        )

        self.cleaner = TextCleaner()

        self.label_index = defaultdict(set)

        self.build_label_index()

    def normalize(self, text):
        return " ".join(
            self.cleaner.clean(text)
            .cleaned_text.casefold()
            .split()
        )

    def build_label_index(self):
        for skill_id, skill in self.repository.items():
            labels = set(
                skill["labels"].values()
            )

            for aliases in skill["aliases"].values():
                labels.update(aliases)

            for label in labels:
                normalized = self.normalize(label)

                if normalized:
                    self.label_index[
                        normalized
                    ].add(skill_id)

    def resolve_skill_id(self, annotation):
        explicit_id = annotation.get("skill_id")

        if explicit_id:
            return explicit_id

        label = annotation["skill"]

        candidates = self.label_index.get(
            self.normalize(label),
            set()
        )

        if len(candidates) == 1:
            return next(iter(candidates))

        if not candidates:
            return f"gold:{self.normalize(label)}"

        details = [
            {
                "skill_id": skill_id,
                "labels": self.repository[skill_id]["labels"]
            }
            for skill_id in sorted(candidates)
        ]

        raise ValueError(
            f"Ambiguous annotation: {label}. "
            f"Choose an explicit skill_id from: "
            f"{json.dumps(details, ensure_ascii=False)}"
        )

    def get_expected(self, document):
        expected = set()

        for annotation in document["expected"]:
            skill_id = self.resolve_skill_id(
                annotation
            )

            status = annotation["status"]

            if status not in VALID_STATUSES:
                raise ValueError(
                    f"Invalid status: {status}"
                )

            expected.add((skill_id, status))

        statuses_by_skill = defaultdict(set)

        for skill_id, status in expected:
            statuses_by_skill[skill_id].add(status)

        if any(
            len(statuses) > 1
            for statuses in statuses_by_skill.values()
        ):
            raise ValueError(
                "Use one aggregated status per skill"
            )

        return expected

    def get_predicted(self, result):
        predicted = set()

        for skill in result["skills"]:
            status = (
                "asserted"
                if skill["asserted_mentions"] > 0
                else "mentioned"
            )

            predicted.add((
                skill["id"],
                status
            ))

        categories = {
            "negated_skills": "negated",
            "learning_skills": "learning",
            "interest_skills": "interest",
            "conflicting_skills": "conflicting"
        }

        for category, status in categories.items():
            for skill in result[category]:
                predicted.add((
                    skill["id"],
                    status
                ))

        return predicted

    def get_skill_label(self, skill_id):
        if skill_id not in self.repository:
            return skill_id

        labels = self.repository[
            skill_id
        ]["labels"]

        return labels.get(
            "en",
            next(iter(labels.values()))
        )

    def evaluate_document(self, document):
        text = self.cleaner.clean(
            document["text"]
        ).cleaned_text

        expected_context = self.get_expected(
            document
        )

        extraction = self.extractor.extract(
            text,
            language=document["language"]
        )

        predicted_context = self.get_predicted(
            extraction
        )

        expected_skills = {
            skill_id
            for skill_id, _ in expected_context
        }

        predicted_skills = {
            skill_id
            for skill_id, _ in predicted_context
        }

        detection_metrics = evaluate_prediction(
            expected_skills,
            predicted_skills
        )

        context_metrics = evaluate_prediction(
            expected_context,
            predicted_context
        )

        errors = []

        for skill_id in sorted(
            predicted_skills - expected_skills
        ):
            errors.append({
                "document_id": document["id"],
                "language": document["language"],
                "error_type": "false_positive",
                "skill_id": skill_id,
                "skill": self.get_skill_label(skill_id),
                "expected_status": None,
                "predicted_status": dict(
                    predicted_context
                )[skill_id]
            })

        for skill_id in sorted(
            expected_skills - predicted_skills
        ):
            errors.append({
                "document_id": document["id"],
                "language": document["language"],
                "error_type": "false_negative",
                "skill_id": skill_id,
                "skill": self.get_skill_label(skill_id),
                "expected_status": dict(
                    expected_context
                )[skill_id],
                "predicted_status": None
            })

        expected_statuses = dict(expected_context)
        predicted_statuses = dict(predicted_context)

        for skill_id in sorted(
            expected_skills & predicted_skills
        ):
            expected_status = expected_statuses[
                skill_id
            ]

            predicted_status = predicted_statuses[
                skill_id
            ]

            if expected_status != predicted_status:
                errors.append({
                    "document_id": document["id"],
                    "language": document["language"],
                    "error_type": "context_error",
                    "skill_id": skill_id,
                    "skill": self.get_skill_label(skill_id),
                    "expected_status": expected_status,
                    "predicted_status": predicted_status
                })

        return (
            detection_metrics,
            context_metrics,
            errors
        )


def load_dataset(path):
    with path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError(
            "Dataset must be a JSON list"
        )

    ids = [document["id"] for document in documents]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs")

    return documents


def prepare_real_dataset(sample_size, seed=42):
    df = pd.read_csv(LINGUISTIC_REPORT)

    df = df[
        df["status"].eq("success")
        & df["language"].isin(["en", "fr"])
    ]

    df = df.drop_duplicates(
        subset=["source_path"]
    )

    if sample_size <= 0 or sample_size > len(df):
        raise ValueError("Invalid sample size")

    rng = random.Random(seed)

    groups = []

    for _, group in df.groupby("category"):
        rows = list(
            group.to_dict(orient="records")
        )

        rng.shuffle(rows)
        groups.append(rows)

    rng.shuffle(groups)

    selected = []

    while len(selected) < sample_size:
        for group in groups:
            if group:
                selected.append(group.pop())

            if len(selected) == sample_size:
                break

    documents = []

    for index, row in enumerate(
        selected,
        start=1
    ):
        path = CLEANED_DIR / row["source_path"]

        text = path.read_text(encoding="utf-8")

        documents.append({
            "id": f"REAL_{index:03d}",
            "language": row["language"],
            "category": row["category"],
            "source_path": row["source_path"],
            "text": text,
            "expected": None
        })

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if REAL_DATASET.exists():
        raise FileExistsError(
            "Annotation dataset already exists. "
            "Move it before generating another sample."
        )

    with REAL_DATASET.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Documents selected: {len(documents)}")
    print(f"Dataset saved to: {REAL_DATASET}")
    print("Manual annotation required.")


def evaluate_dataset(dataset_path):
    documents = load_dataset(dataset_path)

    annotated = [
        document
        for document in documents
        if document.get("expected") is not None
    ]

    if not annotated:
        print("[ERROR] No annotated documents")
        return 1

    output_dir = (
        REPORT_DIR / dataset_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluator = SkillEvaluator()

    detection_results = []
    context_results = []

    language_results = defaultdict(
        lambda: {
            "detection": [],
            "context": []
        }
    )

    records = []
    errors = []
    failed_documents = []

    start = perf_counter()

    for document in annotated:
        document_id = document["id"]

        try:
            detection, context, document_errors = (
                evaluator.evaluate_document(document)
            )

            detection_results.append(detection)
            context_results.append(context)

            language = document["language"]

            language_results[language][
                "detection"
            ].append(detection)

            language_results[language][
                "context"
            ].append(context)

            errors.extend(document_errors)

            record = {
                "document_id": document_id,
                "language": language
            }

            for key, value in (
                detection.to_dict().items()
            ):
                record[f"detection_{key}"] = value

            for key, value in (
                context.to_dict().items()
            ):
                record[f"context_{key}"] = value

            records.append(record)

            print(
                f"[OK] {document_id} | "
                f"Detection F1: {detection.f1_score:.2%} | "
                f"Context F1: {context.f1_score:.2%}"
            )

        except Exception as error:
            failed_documents.append({
                "document_id": document_id,
                "error": str(error)
            })

            print(
                f"[ERROR] {document_id}: {error}"
            )

    summary = {
        "dataset": dataset_path.name,
        "dataset_documents": len(documents),
        "evaluated_documents": len(records),
        "unannotated_documents": (
            len(documents) - len(annotated)
        ),
        "failed_documents": len(failed_documents),
        "complete_evaluation": (
            len(records) == len(documents)
        ),
        "detection": summarize_metrics(
            detection_results
        ),
        "context": summarize_metrics(
            context_results
        ),
        "execution_time": round(
            perf_counter() - start,
            4
        )
    }

    language_records = []

    for language, metrics in language_results.items():
        for evaluation_type in [
            "detection",
            "context"
        ]:
            result = summarize_metrics(
                metrics[evaluation_type]
            )

            language_records.append({
                "language": language,
                "evaluation_type": evaluation_type,
                "documents": result["total_documents"],
                **result["micro_average"],
                "macro_f1": result[
                    "macro_average"
                ]["f1_score"],
                "exact_match_accuracy": result[
                    "exact_match_accuracy"
                ]
            })

    pd.DataFrame(records).to_csv(
        output_dir / "document_metrics.csv",
        index=False
    )

    pd.DataFrame(
        errors,
        columns=[
            "document_id",
            "language",
            "error_type",
            "skill_id",
            "skill",
            "expected_status",
            "predicted_status"
        ]
    ).to_csv(
        output_dir / "extraction_errors.csv",
        index=False
    )

    pd.DataFrame(language_records).to_csv(
        output_dir / "language_metrics.csv",
        index=False
    )

    with (
        output_dir / "evaluation_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    with (
        output_dir / "failed_documents.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            failed_documents,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nSKILL EVALUATION SUMMARY")
    print(json.dumps(summary, indent=4))

    print("\nReports directory:")
    print(output_dir)

    return 1 if not summary["complete_evaluation"] else 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--prepare-review",
        type=int,
        default=None
    )

    args = parser.parse_args()

    if args.prepare_review is not None:
        prepare_real_dataset(
            args.prepare_review
        )

        return 0

    return evaluate_dataset(
        args.dataset
    )


if __name__ == "__main__":
    raise SystemExit(main())
