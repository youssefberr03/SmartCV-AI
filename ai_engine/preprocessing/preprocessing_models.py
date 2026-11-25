
from dataclasses import dataclass, asdict


@dataclass
class PreprocessingResult:
    cleaned_text: str
    original_character_count: int
    cleaned_character_count: int
    original_word_count: int
    cleaned_word_count: int
    removed_characters: int
    reduction_percentage: float
    is_empty: bool

    def to_dict(self):
        return asdict(self)
