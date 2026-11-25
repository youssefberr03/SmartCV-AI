
import pytest

from ai_engine.nlp.skill_extractor import SkillExtractor


def create_skill(skill_id, english, french):
    return {
        "id": skill_id,
        "source": "test",
        "labels": {
            "en": english,
            "fr": french
        },
        "aliases": {
            "en": [english],
            "fr": [french]
        }
    }


@pytest.fixture(scope="module")
def extractor():
    repository = {
        "java": create_skill(
            "java", "Java", "Java"
        ),
        "python": create_skill(
            "python", "Python", "Python"
        ),
        "docker": create_skill(
            "docker", "Docker", "Docker"
        ),
        "sql": create_skill(
            "sql", "SQL", "SQL"
        ),
        "history": create_skill(
            "history", "History", "Histoire"
        ),
        "english": create_skill(
            "english", "English", "Anglais"
        ),
        "machine_learning": create_skill(
            "machine_learning",
            "Machine Learning",
            "Apprentissage automatique"
        ),
        "cpp": create_skill(
            "cpp", "C++", "C++"
        )
    }

    return SkillExtractor(repository)


def get_ids(result, category):
    return {
        skill["id"]
        for skill in result[category]
    }


@pytest.mark.parametrize(
    "text, language",
    [
        ("Experienced in Python.", "en"),
        ("Proficient in Python.", "en"),
        ("Knowledge of Python.", "en"),
        ("I worked with Python.", "en"),
        ("Je maîtrise Python.", "fr"),
        ("Expérience en Python.", "fr")
    ]
)
def test_asserted_skill(
    extractor,
    text,
    language
):
    result = extractor.extract(text, language)

    assert result["skill_count"] == 1
    assert result["skills"][0]["id"] == "python"

    assert (
        result["skills"][0]["asserted_mentions"]
        == 1
    )


@pytest.mark.parametrize(
    "text, language",
    [
        ("No experience with Python.", "en"),
        ("No prior experience with Python.", "en"),
        ("Not familiar with Python.", "en"),
        ("Never used Python.", "en"),
        ("I don't know Python.", "en"),
        ("Aucune expérience en Python.", "fr"),
        ("Pas d'expérience avec Python.", "fr"),
        ("Je ne maîtrise pas Python.", "fr")
    ]
)
def test_negation(
    extractor,
    text,
    language
):
    result = extractor.extract(text, language)

    assert result["skill_count"] == 0

    assert get_ids(
        result, "negated_skills"
    ) == {"python"}


def test_multiple_negated_skills(extractor):
    result = extractor.extract(
        "No experience with Java or Python."
    )

    assert result["skill_count"] == 0

    assert get_ids(
        result, "negated_skills"
    ) == {"java", "python"}


def test_multiple_negated_skills_french(extractor):
    result = extractor.extract(
        "Aucune expérience en Java, Python ou SQL.",
        language="fr"
    )

    assert get_ids(
        result, "negated_skills"
    ) == {"java", "python", "sql"}


def test_negation_scope(extractor):
    result = extractor.extract(
        "No experience with Java or Python, "
        "but experienced in SQL."
    )

    assert get_ids(
        result, "negated_skills"
    ) == {"java", "python"}

    assert get_ids(
        result, "skills"
    ) == {"sql"}


def test_sentence_boundary(extractor):
    result = extractor.extract(
        "No experience with Java. "
        "Experienced in Python."
    )

    assert get_ids(
        result, "negated_skills"
    ) == {"java"}

    assert get_ids(
        result, "skills"
    ) == {"python"}


def test_learning_status(extractor):
    result = extractor.extract(
        "Currently learning Docker."
    )

    assert result["skill_count"] == 0

    assert get_ids(
        result, "learning_skills"
    ) == {"docker"}


def test_multiple_learning_skills(extractor):
    result = extractor.extract(
        "Currently learning Java and Python."
    )

    assert get_ids(
        result, "learning_skills"
    ) == {"java", "python"}


def test_learning_status_french(extractor):
    result = extractor.extract(
        "En cours d'apprentissage de Python.",
        language="fr"
    )

    assert get_ids(
        result, "learning_skills"
    ) == {"python"}


def test_interest_status(extractor):
    result = extractor.extract(
        "Interested in Machine Learning."
    )

    assert result["skill_count"] == 0

    assert get_ids(
        result, "interest_skills"
    ) == {"machine_learning"}


def test_interest_status_french(extractor):
    result = extractor.extract(
        "Intéressé par l'apprentissage automatique.",
        language="fr"
    )

    assert get_ids(
        result, "interest_skills"
    ) == {"machine_learning"}


def test_conflicting_skill(extractor):
    result = extractor.extract(
        "Experienced in Docker. "
        "No experience with Docker."
    )

    assert result["skill_count"] == 0

    assert get_ids(
        result, "conflicting_skills"
    ) == {"docker"}


@pytest.mark.parametrize(
    "text",
    [
        "Employment history",
        "Professional history",
        "Work history",
        "Medical history",
        "Career history"
    ]
)
def test_history_false_positive(extractor, text):
    result = extractor.extract(text)

    assert result["skill_count"] == 0
    assert result["rejected_mention_count"] == 1


def test_valid_history_skill(extractor):
    result = extractor.extract(
        "Degree in history."
    )

    assert get_ids(
        result, "skills"
    ) == {"history"}


def test_native_language(extractor):
    result = extractor.extract(
        "Native language: English."
    )

    assert get_ids(
        result, "skills"
    ) == {"english"}

    assert (
        result["skills"][0]["asserted_mentions"]
        == 1
    )


def test_generic_mention(extractor):
    result = extractor.extract(
        "Python documentation."
    )

    assert result["skill_count"] == 1

    skill = result["skills"][0]

    assert skill["asserted_mentions"] == 0
    assert skill["mentioned_mentions"] == 1


def test_technical_skill(extractor):
    result = extractor.extract(
        "Experienced in C++."
    )

    assert get_ids(
        result, "skills"
    ) == {"cpp"}


def test_new_positive_context(extractor):
    result = extractor.extract(
        "No experience with Java, "
        "experienced in Python."
    )

    assert get_ids(
        result, "negated_skills"
    ) == {"java"}

    assert get_ids(
        result, "skills"
    ) == {"python"}


def test_character_offsets(extractor):
    text = "Currently learning Docker."

    result = extractor.extract(text)

    mention = result[
        "learning_skills"
    ][0]["mentions"][0]

    assert text[
        mention["start_char"]:mention["end_char"]
    ] == "Docker"


def test_context_distribution(extractor):
    result = extractor.extract(
        "Experienced in Java. "
        "Currently learning Docker. "
        "No experience with Python."
    )

    distribution = result[
        "context_distribution"
    ]

    assert distribution["asserted"] == 1
    assert distribution["learning"] == 1
    assert distribution["negated"] == 1


def test_empty_text(extractor):
    result = extractor.extract("")

    assert result["skill_count"] == 0
    assert result["rejected_mention_count"] == 0
