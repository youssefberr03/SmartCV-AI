
from dataclasses import dataclass, asdict
from statistics import mean


@dataclass(frozen=True)
class EvaluationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    exact_match: bool

    def to_dict(self):
        return asdict(self)


def safe_divide(numerator, denominator):
    return (
        numerator / denominator
        if denominator
        else 0.0
    )


def calculate_scores(tp, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def evaluate_prediction(expected, predicted):
    expected = set(expected)
    predicted = set(predicted)

    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)

    scores = calculate_scores(tp, fp, fn)

    return EvaluationMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=scores["precision"],
        recall=scores["recall"],
        f1_score=scores["f1_score"],
        exact_match=expected == predicted
    )


def calculate_micro_average(results):
    tp = sum(
        result.true_positives
        for result in results
    )

    fp = sum(
        result.false_positives
        for result in results
    )

    fn = sum(
        result.false_negatives
        for result in results
    )

    scores = calculate_scores(tp, fp, fn)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        **scores
    }


def calculate_macro_average(results):
    eligible = [
        result
        for result in results
        if (
            result.true_positives
            + result.false_positives
            + result.false_negatives
        ) > 0
    ]

    if not eligible:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "evaluated_documents": 0
        }

    return {
        "precision": mean(
            result.precision
            for result in eligible
        ),
        "recall": mean(
            result.recall
            for result in eligible
        ),
        "f1_score": mean(
            result.f1_score
            for result in eligible
        ),
        "evaluated_documents": len(eligible)
    }


def calculate_exact_match_accuracy(results):
    return safe_divide(
        sum(result.exact_match for result in results),
        len(results)
    )


def summarize_metrics(results):
    results = list(results)

    return {
        "total_documents": len(results),
        "micro_average": calculate_micro_average(results),
        "macro_average": calculate_macro_average(results),
        "exact_match_accuracy":
            calculate_exact_match_accuracy(results)
    }
