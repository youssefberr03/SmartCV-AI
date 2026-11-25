
import pytest

from ai_engine.nlp.linguistic_analyzer import (
    LinguisticAnalyzer
)


@pytest.fixture(scope="module")
def analyzer():
    return LinguisticAnalyzer()


def test_english_detection(analyzer):
    text = (
        "I am a software developer with experience "
        "in building web applications."
    )

    assert analyzer.detect_language(text) == "en"


def test_french_detection(analyzer):
    text = (
        "Je suis développeur informatique et "
        "je travaille sur des applications web."
    )

    assert analyzer.detect_language(text) == "fr"


def test_english_analysis(analyzer):
    result = analyzer.analyze(
        "Developers are building applications.",
        language="en"
    )

    assert result.language == "en"
    assert result.token_count > 0
    assert result.sentence_count == 1


def test_english_lemmatization(analyzer):
    result = analyzer.analyze(
        "They are developing applications.",
        language="en"
    )

    lemmas = [token.lemma for token in result.tokens]

    assert "develop" in lemmas
    assert "application" in lemmas


def test_french_analysis(analyzer):
    result = analyzer.analyze(
        "Les développeurs créent des applications.",
        language="fr"
    )

    assert result.language == "fr"
    assert result.token_count > 0


@pytest.mark.parametrize(
    "skill",
    [
        "C++",
        "C#",
        ".NET",
        "Node.js",
        "React.js",
        "CI/CD"
    ]
)
def test_technical_token_preservation(analyzer, skill):
    result = analyzer.analyze(
        skill,
        language="en"
    )

    assert result.token_count == 1
    assert result.tokens[0].text == skill


def test_stop_words(analyzer):
    result = analyzer.analyze(
        "The developer uses Python.",
        language="en"
    )

    assert result.stop_word_count > 0

    assert "the" not in (
        result.lemmatized_text.split()
    )


def test_negation_preservation(analyzer):
    result = analyzer.analyze(
        "No experience with Java.",
        language="en"
    )

    assert "no" in result.lemmatized_text.split()


def test_pos_tagging(analyzer):
    result = analyzer.analyze(
        "Developers build applications.",
        language="en"
    )

    assert "NOUN" in result.pos_distribution
    assert "VERB" in result.pos_distribution


def test_token_offsets(analyzer):
    text = "Java and Python."

    result = analyzer.analyze(
        text,
        language="en"
    )

    for token in result.tokens:
        assert text[
            token.start_char:token.end_char
        ] == token.text


def test_empty_text(analyzer):
    with pytest.raises(ValueError):
        analyzer.analyze("")


def test_invalid_input(analyzer):
    with pytest.raises(TypeError):
        analyzer.analyze(None)


def test_unsupported_language(analyzer):
    with pytest.raises(ValueError):
        analyzer.analyze(
            "Example text",
            language="unknown"
        )


def test_model_cache(analyzer):
    first = analyzer.load_model("en")
    second = analyzer.load_model("en")

    assert first is second


def test_batch_processing(analyzer):
    documents = [
        ("Developers build applications.", {"id": 1}),
        ("Python is a programming language.", {"id": 2})
    ]

    results = list(
        analyzer.analyze_batch(
            documents,
            language="en"
        )
    )

    assert len(results) == 2
    assert results[0][1]["id"] == 1
    assert results[1][1]["id"] == 2
