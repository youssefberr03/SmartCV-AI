
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def tokenize(text):
    if not isinstance(text, str):
        return []

    return re.findall(r"\b\w+\b", text.lower())


def calculate_lexical_diversity(tokens):
    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def calculate_document_statistics(texts):
    statistics = []

    for text in texts:
        tokens = tokenize(text)

        statistics.append({
            "character_count": len(text),
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens)),
            "lexical_diversity": calculate_lexical_diversity(tokens)
        })

    return pd.DataFrame(statistics)


def calculate_word_frequencies(texts, remove_stop_words=False):
    frequencies = Counter()

    for text in texts:
        tokens = tokenize(text)

        if remove_stop_words:
            tokens = [
                token for token in tokens
                if token not in ENGLISH_STOP_WORDS
                and token.isalpha()
            ]

        frequencies.update(tokens)

    return frequencies


def generate_ngrams(tokens, n):
    if n < 1:
        raise ValueError("n must be greater than zero")

    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def calculate_ngram_frequencies(texts, n):
    frequencies = Counter()

    for text in texts:
        tokens = tokenize(text)
        ngrams = generate_ngrams(tokens, n)

        filtered_ngrams = [
            " ".join(ngram)
            for ngram in ngrams
            if all(
                token not in ENGLISH_STOP_WORDS
                and token.isalpha()
                for token in ngram
            )
        ]

        frequencies.update(filtered_ngrams)

    return frequencies


def calculate_corpus_statistics(texts):
    frequencies = calculate_word_frequencies(texts)

    total_tokens = sum(frequencies.values())
    vocabulary_size = len(frequencies)

    return {
        "document_count": len(texts),
        "total_tokens": total_tokens,
        "vocabulary_size": vocabulary_size,
        "corpus_lexical_diversity": (
            vocabulary_size / total_tokens
            if total_tokens else 0.0
        )
    }
