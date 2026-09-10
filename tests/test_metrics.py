"""
Unit tests for evaluation metrics in hgb_model (Standard unittest, Zero Sklearn).
"""
import unittest
import numpy as np
from hgb_model import (
    compute_confusion_matrix,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_specificity,
    compute_npv,
    compute_f1_score,
    compute_roc_auc,
    compute_roc_curve,
    compute_precision_recall_curve,
)


class TestMetrics(unittest.TestCase):

    def test_confusion_matrix_basic(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
        self.assertEqual(tp, 2)
        self.assertEqual(tn, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
        self.assertEqual(tp + tn + fp + fn, len(y_true))

    def test_metrics_values(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        self.assertTrue(np.isclose(compute_accuracy(y_true, y_pred), 3.0 / 4.0))
        self.assertTrue(np.isclose(compute_precision(y_true, y_pred), 1.0))
        self.assertTrue(np.isclose(compute_recall(y_true, y_pred), 0.5))
        self.assertTrue(np.isclose(compute_specificity(y_true, y_pred), 1.0))
        self.assertTrue(np.isclose(compute_npv(y_true, y_pred), 2.0 / 3.0))
        self.assertTrue(np.isclose(compute_f1_score(y_true, y_pred), 2.0 * 1.0 * 0.5 / (1.0 + 0.5)))

    def test_roc_auc_perfect_and_random(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores_perfect = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertTrue(np.isclose(compute_roc_auc(y_true, y_scores_perfect), 1.0))

        y_scores_worst = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertTrue(np.isclose(compute_roc_auc(y_true, y_scores_worst), 0.0))

        # Ties
        y_scores_tied = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertTrue(np.isclose(compute_roc_auc(y_true, y_scores_tied), 0.5))

    def test_trapezoid_vs_rank_auc(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=500)
        y_scores = rng.rand(500)

        auc_rank = compute_roc_auc(y_true, y_scores)
        fpr, tpr, _ = compute_roc_curve(y_true, y_scores)

        # Trapezoid rule
        auc_trap = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
        self.assertLess(abs(auc_rank - auc_trap), 0.01)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_confusion_matrix([1, 0], [1, 2])

        with self.assertRaises(ValueError):
            compute_roc_auc([1, 0], [np.nan, 0.5])

        with self.assertRaises(ValueError):
            compute_roc_auc([1, 0, 1], [0.5, 0.5])

        with self.assertRaises(ValueError):
            compute_roc_curve([], [])

        with self.assertRaises(ValueError):
            compute_precision_recall_curve([], [])


if __name__ == '__main__':
    unittest.main()
