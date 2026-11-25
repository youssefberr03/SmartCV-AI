
import pytest

from ai_engine.preprocessing.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_unicode_normalization(cleaner):
    result = cleaner.clean("Ｊａｖａ")

    assert result.cleaned_text == "Java"


def test_unicode_composition(cleaner):
    result = cleaner.clean("e\u0301")

    assert result.cleaned_text == "é"


def test_ligature_normalization(cleaner):
    result = cleaner.clean("ﬁnance")

    assert result.cleaned_text == "finance"


def test_multiple_spaces(cleaner):
    result = cleaner.clean("Java     Python    SQL")

    assert result.cleaned_text == "Java Python SQL"


def test_tabs(cleaner):
    result = cleaner.clean("Java\tPython\tSQL")

    assert result.cleaned_text == "Java Python SQL"


def test_line_endings(cleaner):
    result = cleaner.clean("Skills\r\nJava\rPython")

    assert result.cleaned_text == "Skills\nJava\nPython"


def test_multiple_line_breaks(cleaner):
    result = cleaner.clean("Skills\n\n\n\nJava")

    assert result.cleaned_text == "Skills\n\nJava"


def test_invisible_characters(cleaner):
    result = cleaner.clean("\ufeffPy\u200bthon")

    assert result.cleaned_text == "Python"


def test_control_characters(cleaner):
    result = cleaner.clean("Ja\x00va")

    assert result.cleaned_text == "Java"


def test_punctuation_normalization(cleaner):
    result = cleaner.clean("L’expérience “Java”")

    assert result.cleaned_text == "L'expérience \"Java\""


def test_bullet_normalization(cleaner):
    result = cleaner.clean("• Java\n• Python")

    assert result.cleaned_text == "- Java\n- Python"


@pytest.mark.parametrize(
    "skill",
    [
        "C++",
        "C#",
        ".NET",
        "Node.js",
        "React.js",
        "CI/CD",
        "Python 3.13",
        "Bac+5"
    ]
)
def test_technical_skills_preserved(cleaner, skill):
    result = cleaner.clean(skill)

    assert result.cleaned_text == skill


def test_accents_preserved(cleaner):
    text = "Développement et gestion de projet"

    assert cleaner.clean(text).cleaned_text == text


def test_negation_preserved(cleaner):
    text = "No experience in Java"

    assert cleaner.clean(text).cleaned_text == text


def test_empty_text(cleaner):
    result = cleaner.clean("")

    assert result.is_empty is True
    assert result.cleaned_text == ""


def test_whitespace_only(cleaner):
    result = cleaner.clean("   \n\t   ")

    assert result.is_empty is True


def test_invalid_input(cleaner):
    with pytest.raises(TypeError):
        cleaner.clean(None)


def test_statistics(cleaner):
    result = cleaner.clean("Java   Python")

    assert result.original_character_count == 13
    assert result.cleaned_character_count == 11
    assert result.removed_characters == 2
    assert result.cleaned_word_count == 2


def test_idempotence(cleaner):
    text = " Ｊａｖａ   \n\n\n• Python\u200b "

    first_result = cleaner.clean(text).cleaned_text
    second_result = cleaner.clean(first_result).cleaned_text

    assert first_result == second_result
