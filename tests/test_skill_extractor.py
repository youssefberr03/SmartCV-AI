
import pytest

from ai_engine.nlp.skill_extractor import SkillExtractor


def create_skill(skill_id, english, french, aliases):
    return {
        "id": skill_id,
        "source": "test",
        "labels": {
            "en": english,
            "fr": french
        },
        "aliases": aliases
    }


@pytest.fixture(scope="module")
def extractor():
    repository = {
        "java": create_skill(
            "java", "Java", "Java",
            {
                "en": ["Java"],
                "fr": ["Java"]
            }
        ),
        "python": create_skill(
            "python", "Python", "Python",
            {
                "en": ["Python"],
                "fr": ["Python"]
            }
        ),
        "javascript": create_skill(
            "javascript", "JavaScript", "JavaScript",
            {
                "en": ["JavaScript", "JS"],
                "fr": ["JavaScript"]
            }
        ),
        "docker": create_skill(
            "docker", "Docker", "Docker",
            {
                "en": ["Docker"],
                "fr": ["Docker"]
            }
        ),
        "spring_boot": create_skill(
            "spring_boot", "Spring Boot", "Spring Boot",
            {
                "en": ["Spring Boot"],
                "fr": ["Spring Boot"]
            }
        ),
        "project_management": create_skill(
            "project_management",
            "Project management",
            "Gestion de projet",
            {
                "en": ["Project management"],
                "fr": ["Gestion de projet"]
            }
        ),
        "cpp": create_skill(
            "cpp", "C++", "C++",
            {
                "en": ["C++"],
                "fr": ["C++"]
            }
        ),
        "learning": create_skill(
            "learning", "Learning", "Apprentissage",
            {
                "en": ["Learning"],
                "fr": ["Apprentissage"]
            }
        ),
        "machine_learning": create_skill(
            "machine_learning",
            "Machine Learning",
            "Apprentissage automatique",
            {
                "en": ["Machine Learning"],
                "fr": ["Apprentissage automatique"]
            }
        )
    }

    return SkillExtractor(repository)


def get_skill_ids(result):
    return {
        skill["id"]
        for skill in result["skills"]
    }


def test_single_skill(extractor):
    result = extractor.extract("Experience in Java.")

    assert get_skill_ids(result) == {"java"}


def test_multiple_skills(extractor):
    result = extractor.extract("Java, Python and Docker.")

    assert get_skill_ids(result) == {
        "java", "python", "docker"
    }


def test_case_insensitive(extractor):
    result = extractor.extract("JAVA and python")

    assert get_skill_ids(result) == {
        "java", "python"
    }


def test_substring_false_positive(extractor):
    result = extractor.extract("JavaScript developer")

    assert get_skill_ids(result) == {"javascript"}


def test_multiword_skill(extractor):
    result = extractor.extract(
        "Experience with Spring Boot"
    )

    assert "spring_boot" in get_skill_ids(result)


def test_synonym(extractor):
    result = extractor.extract("Experience with JS")

    assert get_skill_ids(result) == {"javascript"}


def test_duplicate_mentions(extractor):
    result = extractor.extract("Java Java Java")

    assert result["skill_count"] == 1
    assert result["skills"][0]["positive_mentions"] == 3


def test_negated_skill(extractor):
    result = extractor.extract(
        "No experience with Docker."
    )

    assert result["skill_count"] == 0
    assert result["negated_skill_count"] == 1


def test_positive_and_negative(extractor):
    result = extractor.extract(
        "No experience with Docker. "
        "Experienced in Java."
    )

    assert get_skill_ids(result) == {"java"}
    assert result["negated_skill_count"] == 1


def test_french_skill(extractor):
    result = extractor.extract(
        "Expérience en gestion de projet.",
        language="fr"
    )

    assert get_skill_ids(result) == {
        "project_management"
    }


def test_bilingual_matching(extractor):
    english = extractor.extract(
        "Project management"
    )

    french = extractor.extract(
        "Gestion de projet",
        language="fr"
    )

    assert get_skill_ids(english) == (
        get_skill_ids(french)
    )


def test_technical_skill(extractor):
    result = extractor.extract("C++ developer")

    assert "cpp" in get_skill_ids(result)


def test_longest_match(extractor):
    result = extractor.extract("Machine Learning")

    assert get_skill_ids(result) == {
        "machine_learning"
    }


def test_no_skill(extractor):
    result = extractor.extract(
        "Hello and welcome."
    )

    assert result["skill_count"] == 0


def test_empty_text(extractor):
    result = extractor.extract("")

    assert result["skill_count"] == 0


def test_invalid_input(extractor):
    with pytest.raises(TypeError):
        extractor.extract(None)


def test_offsets(extractor):
    text = "Experience in Spring Boot."

    result = extractor.extract(text)

    mention = result["skills"][0]["mentions"][0]

    assert text[
        mention["start_char"]:mention["end_char"]
    ] == "Spring Boot"


def test_matcher_cache(extractor):
    first = extractor.get_matcher("en")
    second = extractor.get_matcher("en")

    assert first[1] is second[1]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Java and Python developer",
            {"java", "python"}
        ),
        (
            "JavaScript and Docker",
            {"javascript", "docker"}
        ),
        (
            "Experience with Spring Boot and C++",
            {"spring_boot", "cpp"}
        ),
        (
            "No experience with Docker. Java developer.",
            {"java"}
        ),
        (
            "Machine Learning and Python",
            {"machine_learning", "python"}
        )
    ]
)
def test_annotated_examples(
    extractor,
    text,
    expected
):
    result = extractor.extract(text)

    assert get_skill_ids(result) == expected
