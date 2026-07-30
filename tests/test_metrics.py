import numpy as np

from src.metrics import binary_classification_metrics


def test_false_negative_rate_is_computed_correctly():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.9])

    metrics = binary_classification_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["false_negative_rate"] == 0.5
