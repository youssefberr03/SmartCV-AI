
from pathlib import Path
from collections import Counter
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SKILLS_DIR = (
    BASE_DIR / "data" / "processed" / "skills"
)

REPORT_DIR = (
    BASE_DIR / "reports" / "skills"
)

EXTRACTION_REPORT = (
    REPORT_DIR / "skill_extraction_report.csv"
)


def discover_documents():
    df = pd.read_csv(EXTRACTION_REPORT)

    successful = df[
        df["status"].eq("success")
    ]

    for row in successful.itertuples(index=False):
        path = (
            SKILLS_DIR
            / row.category
            / Path(row.file_name).with_suffix(".json")
        )

        yield path


def audit_context():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    global_distribution = Counter()
    rejected_frequency = Counter()

    records = []
    failed_documents = 0

    for path in discover_documents():
        try:
            with path.open(
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            if "context_distribution" not in data:
                raise ValueError(
                    "Document has not been processed "
                    "with Skill Extractor V3"
                )

            distribution = data[
                "context_distribution"
            ]

            global_distribution.update(
                distribution
            )

            for mention in data[
                "rejected_mentions"
            ]:
                key = (
                    mention["label"],
                    mention["text"].casefold(),
                    mention["reason"]
                )

                rejected_frequency[key] += 1

            records.append({
                "file_name": path.name,
                "category": path.parent.name,
                "positive_skills": data["skill_count"],
                "negated_skills":
                    data["negated_skill_count"],
                "learning_skills":
                    data["learning_skill_count"],
                "interest_skills":
                    data["interest_skill_count"],
                "conflicting_skills":
                    data["conflicting_skill_count"],
                "rejected_mentions":
                    data["rejected_mention_count"],
                "asserted_mentions":
                    distribution.get("asserted", 0),
                "generic_mentions":
                    distribution.get("mentioned", 0)
            })

        except Exception as error:
            failed_documents += 1

            print(
                f"[ERROR] {path.name}: {error}"
            )

    if not records:
        print("[ERROR] No V3 documents available")
        return 1

    df = pd.DataFrame(records)

    df.to_csv(
        REPORT_DIR / "context_report.csv",
        index=False
    )

    summary = {
        "analyzed_documents": len(df),
        "failed_documents": failed_documents,
        "positive_document_skill_pairs": int(
            df["positive_skills"].sum()
        ),
        "negated_document_skill_pairs": int(
            df["negated_skills"].sum()
        ),
        "learning_document_skill_pairs": int(
            df["learning_skills"].sum()
        ),
        "interest_document_skill_pairs": int(
            df["interest_skills"].sum()
        ),
        "conflicting_document_skill_pairs": int(
            df["conflicting_skills"].sum()
        ),
        "rejected_mentions": int(
            df["rejected_mentions"].sum()
        ),
        "documents_with_negations": int(
            df["negated_skills"].gt(0).sum()
        ),
        "documents_with_learning_skills": int(
            df["learning_skills"].gt(0).sum()
        ),
        "documents_with_interests": int(
            df["interest_skills"].gt(0).sum()
        ),
        "documents_with_conflicts": int(
            df["conflicting_skills"].gt(0).sum()
        ),
        "context_distribution": dict(
            global_distribution
        )
    }

    with (
        REPORT_DIR / "context_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    distribution_df = pd.DataFrame(
        global_distribution.most_common(),
        columns=["status", "occurrences"]
    )

    distribution_df.to_csv(
        REPORT_DIR / "context_distribution.csv",
        index=False
    )

    rejected_records = [
        {
            "skill": skill,
            "matched_text": text,
            "reason": reason,
            "occurrences": count
        }
        for (skill, text, reason), count
        in rejected_frequency.most_common()
    ]

    rejected_df = pd.DataFrame(
        rejected_records,
        columns=[
            "skill",
            "matched_text",
            "reason",
            "occurrences"
        ]
    )

    rejected_df.to_csv(
        REPORT_DIR / "context_rejections.csv",
        index=False
    )

    print("\nCONTEXT ANALYSIS SUMMARY")
    print(json.dumps(summary, indent=4))

    print("\nCONTEXT DISTRIBUTION")
    print(distribution_df.to_string(index=False))

    print("\nTOP CONTEXT REJECTIONS")
    print(rejected_df.head(20).to_string(index=False))

    return 1 if failed_documents else 0


if __name__ == "__main__":
    raise SystemExit(audit_context())
