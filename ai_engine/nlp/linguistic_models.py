
from dataclasses import dataclass, field, asdict


@dataclass
class TokenInformation:
    text: str
    lemma: str
    pos: str
    tag: str
    dependency: str
    is_stop: bool
    is_punctuation: bool
    is_number: bool
    start_char: int
    end_char: int


@dataclass
class EntityInformation:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class LinguisticResult:
    language: str
    token_count: int
    unique_token_count: int
    stop_word_count: int
    punctuation_count: int
    sentence_count: int
    vocabulary_size: int
    lemmatized_text: str
    tokens: list[TokenInformation] = field(default_factory=list)
    entities: list[EntityInformation] = field(default_factory=list)
    pos_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
