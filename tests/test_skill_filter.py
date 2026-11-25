
import pytest

from ai_engine.nlp.skill_filter import SkillFilter
from ai_engine.nlp.skill_extractor import SkillExtractor


@pytest.fixture
def skill_filter():
    return SkillFilter()


def create_skill(label, aliases=None):
    return {
        "id": label,
        "source": "test",
        "labels": {
            "en": label
        },
        "aliases": {
            "en": aliases or [label]
        }
    }


@pytest.mark.parametrize(
    "alias, label",
    [
        ("current", "electricity principles"),
        ("equipment", "compile airport certification manuals"),
        ("data", "statistics"),
        ("reporting", "journalism"),
        ("engineering", "packaging engineering"),
        ("processes", "perform ground-handling maintenance procedures"),
        ("security", "securities"),
        ("board", "scaffolding components"),
        ("and procedures", "compile airport certification manuals")
    ]
)
def test_suspicious_alias_rejection(
    skill_filter,
    alias,
    label
):
    skill = create_skill(label)

    assert not skill_filter.is_valid(
        alias,
        skill
    )


@pytest.mark.parametrize(
    "term",
    [
        "Java",
        "Python",
        "C++",
        "C#",
        "SQL",
        "Docker",
        "Spring Boot",
        "PostgreSQL",
        "Communication",
        "Accounting",
        "Logistics",
        "Customer service"
    ]
)
def test_valid_skill_preservation(
    skill_filter,
    term
):
    skill = create_skill(term)

    assert skill_filter.is_valid(
        term,
        skill
    )


def test_unverified_single_word_alias(skill_filter):
    skill = create_skill(
        "Industrial equipment maintenance"
    )

    assert (
        skill_filter.get_rejection_reason(
            "maintenance",
            skill
        )
        == "unverified_single_word_alias"
    )


def test_weak_expression(skill_filter):
    skill = create_skill("Example skill")

    assert (
        skill_filter.get_rejection_reason(
            "and operations",
            skill
        )
        == "weak_expression"
    )


def test_technical_alias(skill_filter):
    skill = create_skill("JavaScript")

    assert skill_filter.is_valid(
        "JS",
        skill
    )


def test_false_positive_regression():
    repository = {
        "electricity": create_skill(
            "electricity principles",
            [
                "electricity principles",
                "current",
                "voltage"
            ]
        ),
        "aviation": create_skill(
            "compile airport certification manuals",
            [
                "compile airport certification manuals",
                "equipment",
                "and procedures"
            ]
        ),
        "statistics": create_skill(
            "statistics",
            [
                "statistics",
                "data"
            ]
        ),
        "java": create_skill("Java")
    }

    extractor = SkillExtractor(repository)

    result = extractor.extract(
        "Current position: Java developer. "
        "Working with data and equipment."
    )

    detected_ids = {
        skill["id"]
        for skill in result["skills"]
    }

    assert detected_ids == {"java"}


def test_specific_concept_preserved():
    repository = {
        "electricity": create_skill(
            "electricity principles",
            [
                "electricity principles",
                "current"
            ]
        )
    }

    extractor = SkillExtractor(repository)

    result = extractor.extract(
        "Knowledge of electricity principles."
    )

    assert result["skill_count"] == 1

    assert (
        result["skills"][0]["id"]
        == "electricity"
    )
