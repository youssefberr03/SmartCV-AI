
import re
import unicodedata

from .preprocessing_models import PreprocessingResult


class TextCleaner:

    CHARACTER_MAPPING = str.maketrans({
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2022": "-",
        "\u200b": "",
        "\ufeff": "",
        "\u00ad": ""
    })

    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKC", text)

    def normalize_line_endings(self, text):
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def normalize_characters(self, text):
        return text.translate(self.CHARACTER_MAPPING)

    def remove_control_characters(self, text):
        return "".join(
            character
            for character in text
            if character in ("\n", "\t")
            or unicodedata.category(character) != "Cc"
        )

    def normalize_whitespace(self, text):
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def clean(self, text):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        original_character_count = len(text)
        original_word_count = len(text.split())

        cleaned_text = self.normalize_unicode(text)
        cleaned_text = self.normalize_line_endings(cleaned_text)
        cleaned_text = self.normalize_characters(cleaned_text)
        cleaned_text = self.remove_control_characters(cleaned_text)
        cleaned_text = self.normalize_whitespace(cleaned_text)

        cleaned_character_count = len(cleaned_text)
        cleaned_word_count = len(cleaned_text.split())

        removed_characters = (
            original_character_count - cleaned_character_count
        )

        reduction_percentage = (
            removed_characters / original_character_count * 100
            if original_character_count > 0
            else 0.0
        )

        return PreprocessingResult(
            cleaned_text=cleaned_text,
            original_character_count=original_character_count,
            cleaned_character_count=cleaned_character_count,
            original_word_count=original_word_count,
            cleaned_word_count=cleaned_word_count,
            removed_characters=removed_characters,
            reduction_percentage=round(reduction_percentage, 4),
            is_empty=not bool(cleaned_text)
        )
