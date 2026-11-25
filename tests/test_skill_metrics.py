
import pytest

from ai_engine.evaluation.skill_metrics import (
    safe_divide,
    calculate_scores,
    evaluate_prediction,
    calculate_micro_average,
    calculate_macro_average,
    calculate_exact_match_accuracy,
    summarize_metrics
)


def test_safe_division():
    assert safe_divide(3, 4) == 0.75


def test_division_by_zero():
    assert safe_divide(1, 0) == 0.0


def test_perfect_prediction():
    expected = {"Java", "Python"}
    predicted = {"Java", "Python"}

    result = evaluate_prediction(
        expected,
        predicted
    )

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0
    assert result.exact_match is True


def test_false_positive():
    result = evaluate_prediction(
        {"Java"},
        {"Java", "Docker"}
    )

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_false_negative():
    result = evaluate_prediction(
        {"Java", "Python"},
        {"Java"}
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 1
    assert result.precision == 1.0
    assert result.recall == 0.5


def test_f1_score():
    scores = calculate_scores(
        tp=3,
        fp=1,
        fn=1
    )

    assert scores["precision"] == 0.75
    assert scores["recall"] == 0.75
    assert scores["f1_score"] == 0.75


def test_no_correct_prediction():
    result = evaluate_prediction(
        {"Java"},
        {"Docker"}
    )

    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.f1_score == 0.0


def test_empty_prediction():
    result = evaluate_prediction(
        {"Java", "Python"},
        set()
    )

    assert result.false_negatives == 2
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_empty_reference():
    result = evaluate_prediction(
        set(),
        {"Java"}
    )

    assert result.false_positives == 1
    assert result.exact_match is False


def test_both_empty():
    result = evaluate_prediction(
        set(),
        set()
    )

    assert result.exact_match is True
    assert result.f1_score == 0.0


def test_duplicate_predictions():
    result = evaluate_prediction(
        ["Java"],
        ["Java", "Java", "Java"]
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.f1_score == 1.0


def test_micro_average():
    results = [
        evaluate_prediction(
            {"Java", "Python"},
            {"Java", "Python"}
        ),
        evaluate_prediction(
            {"SQL", "Docker"},
            {"SQL", "Kubernetes"}
        )
    ]

    summary = calculate_micro_average(results)

    assert summary["true_positives"] == 3
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1

    assert summary["precision"] == 0.75
    assert summary["recall"] == 0.75
    assert summary["f1_score"] == 0.75


def test_macro_average():
    results = [
        evaluate_prediction(
            {"Java"},
            {"Java"}
        ),
        evaluate_prediction(
            {"Python", "SQL", "Docker"},
            {"Python"}
        )
    ]

    summary = calculate_macro_average(results)

    assert summary["precision"] == 1.0
    assert summary["recall"] == pytest.approx(
        2 / 3
    )

    assert summary["f1_score"] == 0.75


def test_macro_empty_document_exclusion():
    results = [
        evaluate_prediction(
            {"Java"},
            {"Java"}
        ),
        evaluate_prediction(
            set(),
            set()
        )
    ]

    summary = calculate_macro_average(results)

    assert summary["evaluated_documents"] == 1
    assert summary["f1_score"] == 1.0


def test_exact_match_accuracy():
    results = [
        evaluate_prediction({"Java"}, {"Java"}),
        evaluate_prediction({"Python"}, {"SQL"}),
        evaluate_prediction(set(), set())
    ]

    accuracy = calculate_exact_match_accuracy(
        results
    )

    assert accuracy == pytest.approx(2 / 3)


def test_context_correct():
    expected = {
        ("docker", "learning")
    }

    predicted = {
        ("docker", "learning")
    }

    result = evaluate_prediction(
        expected,
        predicted
    )

    assert result.f1_score == 1.0


def test_context_incorrect():
    expected = {
        ("docker", "learning")
    }

    predicted = {
        ("docker", "mentioned")
    }

    result = evaluate_prediction(
        expected,
        predicted
    )

    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_empty_summary():
    summary = summarize_metrics([])

    assert summary["total_documents"] == 0
    assert summary["exact_match_accuracy"] == 0.0

    assert (
        summary["micro_average"]["f1_score"]
        == 0.0
    )


def test_complete_summary():
    results = [
        evaluate_prediction(
            {"Java"},
            {"Java"}
        )
    ]

    summary = summarize_metrics(results)

    assert summary["total_documents"] == 1

    assert (
        summary["micro_average"]["f1_score"]
        == 1.0
    )

    assert (
        summary["macro_average"]["f1_score"]
        == 1.0
    )

    assert summary["exact_match_accuracy"] == 1.0
