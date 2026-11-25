
import json

from ai_engine.nlp.skill_repository import SkillRepository
from ai_engine.nlp.skill_extractor import SkillExtractor


TECHNICAL_TERMS = [
    "SQL",
    "C#",
    "C++",
    "Java",
    "Python",
    "JavaScript",
    "PostgreSQL",
    "Docker",
    "Git",
    "Node.js",
    ".NET",
    "React.js",
    "Spring Boot",
    "Machine Learning"
]


def audit_technical_skills():
    repository = SkillRepository().load()

    extractor = SkillExtractor(repository)

    for term in TECHNICAL_TERMS:

        print("\n" + "=" * 60)
        print(f"TERM: {term}")
        print("=" * 60)

        candidates = []

        for skill_id, skill in repository.items():

            labels = set(
                skill["labels"].values()
            )

            for aliases in skill["aliases"].values():
                labels.update(aliases)

            normalized_labels = {
                label.casefold()
                for label in labels
            }

            if term.casefold() in normalized_labels:

                candidates.append({
                    "id": skill_id,
                    "labels": skill["labels"],
                    "source": skill["source"]
                })

        print("\nREPOSITORY CANDIDATES")

        print(json.dumps(
            candidates,
            ensure_ascii=False,
            indent=2
        ))

        examples = {
            "en": f"Experienced in {term}.",
            "fr": f"Je maîtrise {term}."
        }

        for language, text in examples.items():

            nlp, matcher = extractor.get_matcher(
                language
            )

            doc = nlp.make_doc(text)

            result = extractor.extract(
                text,
                language
            )

            print(f"\nLANGUAGE: {language}")
            print(f"TEXT: {text}")

            print("TOKENS:", [
                token.text
                for token in doc
            ])

            print("RAW MATCHES:", [
                {
                    "skill_id": nlp.vocab.strings[
                        match_id
                    ],
                    "text": doc[start:end].text
                }
                for match_id, start, end
                in matcher(doc)
            ])

            print("DETECTED SKILLS:", [
                {
                    "id": skill["id"],
                    "label": skill["label"],
                    "mentions": skill["mentions"]
                }
                for skill in result["skills"]
            ])


if __name__ == "__main__":
    audit_technical_skills()
