from pathlib import Path
from collections import Counter
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SKILLS_DIR = BASE_DIR / "data" / "processed" / "skills"
REPORT_DIR = BASE_DIR / "reports" / "skills"

SUSPICIOUS_SKILLS = {
    "electricity principles",
    "compile airport certification manuals",
    "perform ground-handling maintenance procedures",
    "scaffolding components",
    "securities",
    "statistics",
    "packaging engineering",
    "journalism"
}


def audit_skills():
    mentions = Counter()
    document_counts = Counter()

    for path in SKILLS_DIR.rglob("*.json"):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for skill in data["skills"]:
            label = skill["label"].casefold()

            if label not in SUSPICIOUS_SKILLS:
                continue

            document_counts[label] += 1

            for mention in skill["mentions"]:
                if mention["negated"]:
                    continue

                key = (
                    label,
                    mention["text"].casefold()
                )

                mentions[key] += 1

    records = [
        {
            "skill": skill,
            "matched_text": text,
            "occurrences": count
        }
        for (skill, text), count
        in mentions.most_common()
    ]

    df = pd.DataFrame(
        records,
        columns=[
            "skill",
            "matched_text",
            "occurrences"
        ]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        REPORT_DIR / "skill_audit.csv",
        index=False
    )

    for skill in sorted(SUSPICIOUS_SKILLS):
        print(f"\n{skill.upper()}")
        print(f"Documents: {document_counts[skill]}")

        subset = df[df["skill"].eq(skill)]

        print(
            subset[
                ["matched_text", "occurrences"]
            ].head(15).to_string(index=False)
        )


if __name__ == "__main__":
    audit_skills()