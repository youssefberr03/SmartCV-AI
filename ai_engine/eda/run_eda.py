
from time import perf_counter

from .corpus_analysis import (
    analyze_resumes,
    analyze_jobs
)

from .reference_analysis import (
    analyze_esco,
    analyze_matching
)


def main():
    print("SMARTCV AI - Exploratory Data Analysis")

    analyses = [
        ("Resume Dataset", analyze_resumes),
        ("Job Postings", analyze_jobs),
        ("ESCO", analyze_esco),
        ("Candidate Matching", analyze_matching)
    ]

    global_start = perf_counter()
    results = {}

    for name, analysis in analyses:
        start = perf_counter()

        try:
            analysis()

            elapsed = perf_counter() - start

            results[name] = True

            print(f"\n[OK] {name}: {elapsed:.2f} seconds")

        except Exception as error:
            results[name] = False

            print(f"\n[ERROR] {name}: {error}")

    print("\nEDA SUMMARY")

    for name, success in results.items():
        status = "OK" if success else "FAILED"

        print(f"[{status}] {name}")

    total_time = perf_counter() - global_start

    print(f"\nTotal execution time: {total_time:.2f} seconds")

    if not all(results.values()):
        return 1

    print("Exploratory data analysis completed successfully!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
