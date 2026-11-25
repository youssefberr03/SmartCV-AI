
import pytest
import spacy

from ai_engine.nlp.technical_tokenizer import (
    configure_technical_tokenizer
)

from ai_engine.nlp.skill_extractor import (
    SkillExtractor
)


@pytest.fixture(scope="module", params=["en", "fr"])
def nlp(request):
    return configure_technical_tokenizer(
        spacy.blank(request.param)
    )


@pytest.mark.parametrize(
    "term",
    [
        "C#",
        "C++",
        ".NET",
        "Node.js",
        "React.js",
        "Vue.js",
        "CI/CD",
        "ASP.NET"
    ]
)
def test_technical_term_preservation(nlp, term):
    doc = nlp(term)

    assert [token.text for token in doc] == [term]


@pytest.mark.parametrize(
    "punctuation",
    [".", ",", ";", ":", "!", "?"]
)
def test_csharp_with_punctuation(nlp, punctuation):
    doc = nlp("C#" + punctuation)

    assert [token.text for token in doc] == [
        "C#",
        punctuation
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("C++.", ["C++", "."]),
        ("Node.js.", ["Node.js", "."]),
        (".NET,", [".NET", ","]),
        ("React.js;", ["React.js", ";"]),
        ("CI/CD:", ["CI/CD", ":"]),
        ("c#.", ["c#", "."])
    ]
)
def test_technical_punctuation(nlp, text, expected):
    assert [
        token.text for token in nlp(text)
    ] == expected


@pytest.fixture(scope="module")
def extractor():
    repository = {
        "csharp": {
            "id": "csharp",
            "source": "test",
            "labels": {
                "en": "C#",
                "fr": "C#"
            },
            "aliases": {
                "en": ["C#"],
                "fr": ["C#"]
            }
        },
        "cpp": {
            "id": "cpp",
            "source": "test",
            "labels": {
                "en": "C++",
                "fr": "C++"
            },
            "aliases": {
                "en": ["C++"],
                "fr": ["C++"]
            }
        }
    }

    return SkillExtractor(repository)


@pytest.mark.parametrize(
    "language, text",
    [
        ("en", "Experienced in C#."),
        ("fr", "Je maîtrise C#."),
        ("fr", "Je maîtrise c#.")
    ]
)
def test_csharp_extraction(
    extractor,
    language,
    text
):
    result = extractor.extract(
        text,
        language
    )

    assert result["skill_count"] == 1
    assert result["skills"][0]["id"] == "csharp"
    assert result["skills"][0]["asserted_mentions"] == 1


def test_multiple_technical_skills(extractor):
    result = extractor.extract(
        "Je maîtrise C++ et C#.",
        language="fr"
    )

    detected = {
        skill["id"]
        for skill in result["skills"]
    }

    assert detected == {"cpp", "csharp"}


def test_csharp_negation(extractor):
    result = extractor.extract(
        "Aucune expérience en C#.",
        language="fr"
    )

    assert result["skill_count"] == 0

    assert {
        skill["id"]
        for skill in result["negated_skills"]
    } == {"csharp"}


def test_original_character_offsets(extractor):
    text = "Je maîtrise C#."

    result = extractor.extract(
        text,
        language="fr"
    )

    mention = result["skills"][0]["mentions"][0]

    start = mention["start_char"]
    end = mention["end_char"]

    assert text[start:end] == "C#"
