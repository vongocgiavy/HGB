"""
Unit tests for train_test_split_stratified in hgb_model (Standard unittest).
"""
import unittest
import numpy as np
from hgb_model import train_test_split_stratified


class TestSplit(unittest.TestCase):

    def test_stratified_split_proportions(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 5)
        y = np.array([0] * 700 + [1] * 300)

        X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_tr), 800)
        self.assertEqual(len(X_te), 200)
        self.assertEqual(len(y_tr), 800)
        self.assertEqual(len(y_te), 200)

        # Check class ratios
        self.assertTrue(np.isclose(np.mean(y_tr), 0.3, atol=0.01))
        self.assertTrue(np.isclose(np.mean(y_te), 0.3, atol=0.01))

    def test_return_indices_and_overlap(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 4)
        y = rng.randint(0, 2, size=500)

        X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split_stratified(
            X, y, test_size=0.25, random_state=42, return_indices=True
        )
        self.assertEqual(len(tr_idx), 375)
        self.assertEqual(len(te_idx), 125)
        self.assertEqual(len(tr_idx) + len(te_idx), 500)
        self.assertEqual(len(np.intersect1d(tr_idx, te_idx)), 0)
        self.assertTrue(np.array_equal(X[tr_idx], X_tr))
        self.assertTrue(np.array_equal(X[te_idx], X_te))

    def test_invalid_parameters(self):
        X = np.ones((10, 2))
        y = np.array([0, 1] * 5)

        with self.assertRaises(ValueError):
            train_test_split_stratified(X, y, test_size=1.5)

        with self.assertRaises(ValueError):
            train_test_split_stratified(X, y, test_size=0.0)

        with self.assertRaises(ValueError):
            train_test_split_stratified(X, np.zeros(10))

        with self.assertRaises(ValueError):
            train_test_split_stratified(X, np.array([0, 1] * 4))

        with self.assertRaises(ValueError):
            train_test_split_stratified(np.ones((9, 2)), y)


if __name__ == '__main__':
    unittest.main()
