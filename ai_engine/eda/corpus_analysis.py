
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .text_statistics import (
    calculate_document_statistics,
    calculate_word_frequencies,
    calculate_ngram_frequencies,
    calculate_corpus_statistics
)


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "eda"

TOP_N = 25
RANDOM_STATE = 42
JOB_CHUNK_SIZE = 5000
JOB_SAMPLE_FRACTION = 0.08


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def plot_distribution(values, title, xlabel, output_path):
    plt.figure(figsize=(11, 6))

    plt.hist(
        values,
        bins=50,
        color="steelblue",
        edgecolor="white"
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of documents")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_frequencies(frequencies, title, output_path):
    if frequencies.empty:
        return

    frequencies = frequencies.sort_values()

    plt.figure(figsize=(11, 9))
    frequencies.plot(kind="barh", color="steelblue")

    plt.title(title)
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_zipf(frequencies, output_path):
    if not frequencies:
        return

    counts = sorted(frequencies.values(), reverse=True)
    ranks = range(1, len(counts) + 1)

    plt.figure(figsize=(10, 6))

    plt.loglog(
        ranks,
        counts,
        label="Observed frequencies"
    )

    plt.loglog(
        ranks,
        [counts[0] / rank for rank in ranks],
        linestyle="--",
        label="Theoretical Zipf distribution"
    )

    plt.title("Zipf Distribution")
    plt.xlabel("Word rank (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def analyze_texts(texts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not texts:
        raise ValueError("No valid documents available")

    document_statistics = calculate_document_statistics(texts)

    document_statistics.to_csv(
        output_dir / "document_statistics.csv",
        index=False
    )

    document_statistics.describe().to_csv(
        output_dir / "statistical_summary.csv"
    )

    corpus_statistics = calculate_corpus_statistics(texts)

    save_json(
        corpus_statistics,
        output_dir / "corpus_statistics.json"
    )

    raw_frequencies = calculate_word_frequencies(texts)

    filtered_frequencies = calculate_word_frequencies(
        texts,
        remove_stop_words=True
    )

    raw_top_words = pd.Series(
        dict(raw_frequencies.most_common(TOP_N)),
        dtype="int64"
    )

    filtered_top_words = pd.Series(
        dict(filtered_frequencies.most_common(TOP_N)),
        dtype="int64"
    )

    raw_top_words.to_csv(
        output_dir / "top_words_raw.csv",
        header=["frequency"]
    )

    filtered_top_words.to_csv(
        output_dir / "top_words_filtered.csv",
        header=["frequency"]
    )

    plot_frequencies(
        raw_top_words,
        "Most Frequent Words",
        output_dir / "top_words_raw.png"
    )

    plot_frequencies(
        filtered_top_words,
        "Most Frequent Words Without Stop Words",
        output_dir / "top_words_filtered.png"
    )

    plot_distribution(
        document_statistics["token_count"],
        "Document Length Distribution",
        "Number of tokens",
        output_dir / "document_length.png"
    )

    plot_distribution(
        document_statistics["lexical_diversity"],
        "Lexical Diversity Distribution",
        "Type-Token Ratio",
        output_dir / "lexical_diversity.png"
    )

    plot_zipf(
        raw_frequencies,
        output_dir / "zipf_distribution.png"
    )

    for n in [2, 3]:
        frequencies = calculate_ngram_frequencies(texts, n)

        top_ngrams = pd.Series(
            dict(frequencies.most_common(TOP_N)),
            dtype="int64"
        )

        top_ngrams.to_csv(
            output_dir / f"top_{n}grams.csv",
            header=["frequency"]
        )

        plot_frequencies(
            top_ngrams,
            f"Most Frequent {n}-grams",
            output_dir / f"top_{n}grams.png"
        )

    print(json.dumps(corpus_statistics, indent=4))
    print(f"Average tokens: {document_statistics['token_count'].mean():.2f}")
    print(f"Median tokens: {document_statistics['token_count'].median():.2f}")


def analyze_resumes():
    print("\nAnalyzing Resume Dataset...")

    output_dir = REPORT_DIR / "resumes"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        RAW_DIR / "resumes" / "Resume.csv",
        usecols=["Resume_str", "Category"],
        low_memory=False
    )

    categories = df["Category"].value_counts()

    categories.to_csv(
        output_dir / "category_distribution.csv"
    )

    plot_frequencies(
        categories,
        "Resume Category Distribution",
        output_dir / "category_distribution.png"
    )

    texts = [
        text.strip()
        for text in df["Resume_str"].dropna().astype(str)
        if text.strip()
    ]

    print(f"Original rows: {len(df)}")
    print(f"Valid documents: {len(texts)}")
    print(f"Categories: {len(categories)}")

    analyze_texts(texts, output_dir)


def load_job_sample():
    samples = []
    total_rows = 0

    reader = pd.read_csv(
        RAW_DIR / "jobs" / "postings.csv",
        usecols=["title", "description"],
        chunksize=JOB_CHUNK_SIZE
    )

    for index, chunk in enumerate(reader):
        total_rows += len(chunk)

        valid_chunk = chunk.dropna(subset=["description"])

        valid_chunk = valid_chunk[
            valid_chunk["description"].str.strip().ne("")
        ]

        sample = valid_chunk.sample(
            frac=JOB_SAMPLE_FRACTION,
            random_state=RANDOM_STATE + index
        )

        samples.append(sample)

    if not samples:
        raise ValueError("No job postings available")

    return pd.concat(samples, ignore_index=True), total_rows


def analyze_jobs():
    print("\nAnalyzing Job Postings...")

    df, total_rows = load_job_sample()

    output_dir = REPORT_DIR / "jobs"
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = df["description"].astype(str).tolist()

    save_json({
        "original_rows": total_rows,
        "sample_rows": len(df),
        "sampling_fraction": JOB_SAMPLE_FRACTION,
        "random_state": RANDOM_STATE
    }, output_dir / "sampling_information.json")

    print(f"Original rows: {total_rows}")
    print(f"Sampled rows: {len(df)}")

    analyze_texts(texts, output_dir)
