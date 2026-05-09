import unittest

import numpy as np

from src.evaluate import best_cost_threshold, classification_metrics, cost_metrics


class EvaluateTest(unittest.TestCase):
    def test_classification_metrics_counts_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.8, 0.4, 0.9])

        metrics = classification_metrics(y_true, scores, threshold=0.5)

        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["alerts"], 2)

    def test_cost_metrics_weights_false_negatives(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.8, 0.4, 0.9])

        metrics = cost_metrics(
            y_true,
            scores,
            threshold=0.5,
            false_positive_cost=10,
            false_negative_cost=100,
        )

        self.assertEqual(metrics["total_cost"], 110)

    def test_best_cost_threshold_returns_threshold_from_grid(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])

        threshold, curve = best_cost_threshold(
            y_true,
            scores,
            false_positive_cost=1,
            false_negative_cost=100,
        )

        self.assertGreaterEqual(threshold, 0.01)
        self.assertLessEqual(threshold, 0.99)
        self.assertIn("total_cost", curve.columns)


if __name__ == "__main__":
    unittest.main()

