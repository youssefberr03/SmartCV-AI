
from collections import Counter

import spacy
from spacy.symbols import ORTH
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from .linguistic_models import (
    TokenInformation,
    EntityInformation,
    LinguisticResult
)


DetectorFactory.seed = 42


class LinguisticAnalyzer:

    MODELS = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm"
    }

    TECHNICAL_TERMS = [
        "C++",
        "C#",
        ".NET",
        "Node.js",
        "React.js",
        "Vue.js",
        "Next.js",
        "CI/CD",
        "ASP.NET"
    ]

    NEGATION_WORDS = {
        "no", "not", "never", "without", "n't",
        "ne", "pas", "jamais", "sans", "aucun"
    }

    def __init__(self):
        self.models = {}

    def detect_language(self, text):
        if not text.strip():
            return "unknown"

        try:
            language = detect(text[:10000])

            return (
                language
                if language in self.MODELS
                else "unknown"
            )

        except LangDetectException:
            return "unknown"

    def load_model(self, language):
        if language not in self.MODELS:
            raise ValueError(
                f"Unsupported language: {language}"
            )

        if language not in self.models:
            nlp = spacy.load(self.MODELS[language])

            for term in self.TECHNICAL_TERMS:
                for variant in {
                    term,
                    term.lower(),
                    term.upper()
                }:
                    nlp.tokenizer.add_special_case(
                        variant,
                        [{ORTH: variant}]
                    )

            self.models[language] = nlp

        return self.models[language]

    def build_result(self, doc, language):
        tokens = []
        entities = []
        pos_distribution = Counter()
        lemmas = []

        for token in doc:
            if token.is_space:
                continue

            lemma = token.lemma_ or token.text

            if token.text.lower() in {
                term.lower()
                for term in self.TECHNICAL_TERMS
            }:
                lemma = token.text

            tokens.append(
                TokenInformation(
                    text=token.text,
                    lemma=lemma,
                    pos=token.pos_,
                    tag=token.tag_,
                    dependency=token.dep_,
                    is_stop=token.is_stop,
                    is_punctuation=token.is_punct,
                    is_number=token.like_num,
                    start_char=token.idx,
                    end_char=token.idx + len(token.text)
                )
            )

            pos_distribution[token.pos_] += 1

            if token.is_punct:
                continue

            if (
                token.is_stop
                and token.lower_ not in self.NEGATION_WORDS
            ):
                continue

            lemmas.append(lemma.lower())

        for entity in doc.ents:
            entities.append(
                EntityInformation(
                    text=entity.text,
                    label=entity.label_,
                    start_char=entity.start_char,
                    end_char=entity.end_char
                )
            )

        return LinguisticResult(
            language=language,
            token_count=len(tokens),
            unique_token_count=len({
                token.text.lower()
                for token in tokens
            }),
            stop_word_count=sum(
                token.is_stop for token in tokens
            ),
            punctuation_count=sum(
                token.is_punctuation for token in tokens
            ),
            sentence_count=sum(1 for _ in doc.sents),
            vocabulary_size=len(set(lemmas)),
            lemmatized_text=" ".join(lemmas),
            tokens=tokens,
            entities=entities,
            pos_distribution=dict(pos_distribution)
        )

    def analyze(self, text, language=None):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        if not text.strip():
            raise ValueError("Text cannot be empty")

        language = language or self.detect_language(text)

        nlp = self.load_model(language)
        doc = nlp(text)

        return self.build_result(doc, language)

    def analyze_batch(
        self,
        documents,
        language,
        batch_size=32
    ):
        nlp = self.load_model(language)

        for doc, context in nlp.pipe(
            documents,
            as_tuples=True,
            batch_size=batch_size
        ):
            yield (
                self.build_result(doc, language),
                context
            )
