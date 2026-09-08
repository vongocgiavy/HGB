"""
Unit tests for CustomHistGradientBoostingClassifier in hgb_model (Standard unittest).
"""
import unittest
import numpy as np
from hgb_model import CustomHistGradientBoostingClassifier


class TestClassifier(unittest.TestCase):

    def test_classifier_fit_predict_dev_mode(self):
        rng = np.random.RandomState(42)
        X = rng.randn(400, 6).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

        clf = CustomHistGradientBoostingClassifier(
            n_estimators=30,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=10,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42
        )
        clf.fit(X, y)

        self.assertGreater(clf.n_iter_, 0)
        self.assertGreater(clf.best_n_iter_, 0)
        self.assertIsNotNone(clf.X_val_)
        self.assertIsNotNone(clf.y_val_)
        self.assertEqual(len(clf.X_val_), int(np.round(400 * 0.15)))

        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (400,))
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))

        pred = clf.predict(X, threshold=0.5)
        self.assertEqual(pred.shape, (400,))
        self.assertTrue(set(np.unique(pred)).issubset({0, 1}))

    def test_classifier_fit_full_mode(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 4).astype(np.float32)
        y = rng.randint(0, 2, size=300).astype(np.float32)

        # Full mode: validation_fraction = 0.0
        clf = CustomHistGradientBoostingClassifier(
            n_estimators=15,
            learning_rate=0.1,
            max_depth=3,
            validation_fraction=0.0,
            n_iter_no_change=0,
            random_state=42
        )
        clf.fit(X, y)

        self.assertIsNone(clf.X_val_)
        self.assertEqual(clf.n_iter_, 15)
        self.assertEqual(len(clf.trees), 15)

        preds = clf.predict(X)
        self.assertEqual(len(preds), 300)

    def test_unfitted_raises_error(self):
        clf = CustomHistGradientBoostingClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(np.ones((10, 5)))

        with self.assertRaises(RuntimeError):
            clf.predict_proba(np.ones((10, 5)))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            CustomHistGradientBoostingClassifier(n_estimators=0)

        with self.assertRaises(ValueError):
            CustomHistGradientBoostingClassifier(learning_rate=-0.1)

        with self.assertRaises(ValueError):
            CustomHistGradientBoostingClassifier(max_depth=0)

        with self.assertRaises(ValueError):
            CustomHistGradientBoostingClassifier(validation_fraction=1.0)


if __name__ == '__main__':
    unittest.main()
