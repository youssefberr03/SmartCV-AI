
from pathlib import Path
from collections import defaultdict
import json

import pandas as pd

from ai_engine.preprocessing.text_cleaner import TextCleaner


BASE_DIR = Path(__file__).resolve().parents[2]

ESCO_DIR = BASE_DIR / "data" / "raw" / "esco"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "esco"

REPOSITORY_PATH = OUTPUT_DIR / "skill_repository.json"


TECHNICAL_SKILLS = {
    "java": ["Java"],
    "javascript": ["JavaScript", "JS"],
    "python": ["Python"],
    "typescript": ["TypeScript"],
    "cplusplus": ["C++"],
    "csharp": ["C#"],
    "sql": ["SQL"],
    "postgresql": ["PostgreSQL", "Postgres"],
    "mysql": ["MySQL"],
    "mongodb": ["MongoDB"],
    "spring_boot": ["Spring Boot"],
    "spring_framework": ["Spring Framework"],
    "react": ["React.js", "ReactJS"],
    "angular": ["Angular", "AngularJS"],
    "nodejs": ["Node.js", "NodeJS"],
    "docker": ["Docker"],
    "kubernetes": ["Kubernetes", "K8s"],
    "git": ["Git"],
    "github": ["GitHub"],
    "gitlab": ["GitLab"],
    "dotnet": [".NET", "DotNet"],
    "cicd": ["CI/CD"],
    "aws": ["AWS", "Amazon Web Services"],
    "machine_learning": [
        "Machine Learning",
        "Apprentissage automatique"
    ],
    "deep_learning": [
        "Deep Learning",
        "Apprentissage profond"
    ],
    "nlp": [
        "Natural Language Processing",
        "Traitement automatique du langage naturel",
        "NLP"
    ],
    "rest_api": [
        "REST API",
        "REST APIs",
        "API REST"
    ]
}


TECHNICAL_OVERRIDES = {
    "sql": (
        "http://data.europa.eu/esco/skill/"
        "598de5b0-5b58-4ea7-8058-a4bc4d18c742"
    )
}


class SkillRepository:

    def __init__(self):
        self.skills = {}
        self.cleaner = TextCleaner()
        self.technical_conflicts = []

    def normalize_label(self, label):
        if not isinstance(label, str):
            return ""

        return " ".join(
            self.cleaner.clean(label)
            .cleaned_text.split()
        )

    def extract_aliases(self, value):
        if not isinstance(value, str):
            return []

        return [
            self.normalize_label(alias)
            for alias in value.splitlines()
            if alias.strip()
        ]

    def load_esco(self, language):
        path = (
            ESCO_DIR
            / language
            / f"skills_{language}.csv"
        )

        columns = [
            "conceptUri",
            "preferredLabel",
            "altLabels",
            "skillType",
            "status"
        ]

        df = pd.read_csv(
            path,
            usecols=columns
        )

        df = df[
            df["status"].eq("released")
        ].dropna(
            subset=[
                "conceptUri",
                "preferredLabel"
            ]
        )

        for row in df.itertuples(index=False):
            skill_id = row.conceptUri

            label = self.normalize_label(
                row.preferredLabel
            )

            aliases = self.extract_aliases(
                row.altLabels
            )

            if not label:
                continue

            if skill_id not in self.skills:
                self.skills[skill_id] = {
                    "id": skill_id,
                    "source": "esco",
                    "skill_type": row.skillType,
                    "labels": {},
                    "aliases": {}
                }

            skill = self.skills[skill_id]

            skill["labels"][language] = label

            skill["aliases"][language] = sorted(
                set([label, *aliases])
            )

    def build_indexes(self):
        preferred_index = defaultdict(set)
        alias_index = defaultdict(set)

        for skill_id, skill in self.skills.items():

            for label in skill["labels"].values():
                preferred_index[
                    label.casefold()
                ].add(skill_id)

            for aliases in skill["aliases"].values():
                for alias in aliases:
                    alias_index[
                        alias.casefold()
                    ].add(skill_id)

        return preferred_index, alias_index

    def resolve_technical_skill(
        self,
        key,
        canonical_label
    ):
        preferred_index, alias_index = (
            self.build_indexes()
        )

        normalized = canonical_label.casefold()

        if key in TECHNICAL_OVERRIDES:
            skill_id = TECHNICAL_OVERRIDES[key]

            if skill_id not in self.skills:
                raise ValueError(
                    f"Unknown technical override: {skill_id}"
                )

            return skill_id

        preferred_matches = preferred_index.get(
            normalized,
            set()
        )

        alias_matches = alias_index.get(
            normalized,
            set()
        )

        if len(preferred_matches) == 1:
            return next(iter(preferred_matches))

        if len(preferred_matches) > 1:
            candidates = preferred_matches

        elif len(alias_matches) == 1:
            return next(iter(alias_matches))

        elif len(alias_matches) > 1:
            candidates = alias_matches

        else:
            return None

        self.technical_conflicts.append({
            "technical_skill": canonical_label,
            "candidate_ids": sorted(candidates)
        })

        return None

    def remove_sql_alias_conflicts(self):
        target_id = TECHNICAL_OVERRIDES["sql"]

        for skill_id, skill in self.skills.items():

            if skill_id == target_id:
                continue

            preferred_labels = {
                label.casefold()
                for label in skill["labels"].values()
            }

            if "sql" in preferred_labels:
                self.technical_conflicts.append({
                    "technical_skill": "SQL",
                    "candidate_ids": [
                        target_id,
                        skill_id
                    ]
                })

                continue

            for language, aliases in (
                skill["aliases"].items()
            ):
                skill["aliases"][language] = [
                    alias
                    for alias in aliases
                    if alias.casefold() != "sql"
                ]

    def add_technical_skills(self):

        for key, aliases in TECHNICAL_SKILLS.items():

            canonical_label = aliases[0]

            previous_conflicts = len(
                self.technical_conflicts
            )

            skill_id = self.resolve_technical_skill(
                key,
                canonical_label
            )

            if skill_id is None:

                if len(self.technical_conflicts) > (
                    previous_conflicts
                ):
                    continue

                skill_id = f"local:{key}"

                self.skills[skill_id] = {
                    "id": skill_id,
                    "source": "custom",
                    "skill_type": "skill/competence",
                    "labels": {
                        "en": canonical_label,
                        "fr": canonical_label
                    },
                    "aliases": {}
                }

            skill = self.skills[skill_id]

            for language in ["en", "fr"]:

                existing = skill["aliases"].get(
                    language,
                    []
                )

                skill["aliases"][language] = sorted(
                    set(existing + aliases)
                )

        self.remove_sql_alias_conflicts()

    def build(self):
        self.skills = {}
        self.technical_conflicts = []

        for language in ["en", "fr"]:
            self.load_esco(language)

        self.add_technical_skills()

        return self.skills

    def get_statistics(self):
        return {
            "total_concepts": len(self.skills),

            "esco_concepts": sum(
                skill["source"] == "esco"
                for skill in self.skills.values()
            ),

            "custom_concepts": sum(
                skill["source"] == "custom"
                for skill in self.skills.values()
            ),

            "bilingual_concepts": sum(
                "en" in skill["labels"]
                and "fr" in skill["labels"]
                for skill in self.skills.values()
            ),

            "english_aliases": sum(
                len(skill["aliases"].get("en", []))
                for skill in self.skills.values()
            ),

            "french_aliases": sum(
                len(skill["aliases"].get("fr", []))
                for skill in self.skills.values()
            ),

            "technical_conflicts": len(
                self.technical_conflicts
            )
        }

    def save(self):
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with REPOSITORY_PATH.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.skills,
                file,
                ensure_ascii=False,
                indent=2
            )

        with (
            OUTPUT_DIR / "technical_conflicts.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                self.technical_conflicts,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self):
        with REPOSITORY_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:
            self.skills = json.load(file)

        return self.skills


if __name__ == "__main__":
    repository = SkillRepository()

    repository.build()
    repository.save()

    print("\nSKILL REPOSITORY STATISTICS")

    print(
        json.dumps(
            repository.get_statistics(),
            indent=4
        )
    )

    print("\nTECHNICAL CONFLICTS")

    print(
        json.dumps(
            repository.technical_conflicts,
            ensure_ascii=False,
            indent=4
        )
    )

    print(f"\nRepository saved to: {REPOSITORY_PATH}")
