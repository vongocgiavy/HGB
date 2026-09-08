"""
Unit tests for data leakage boundaries and pipeline integrity (Standard unittest).
"""
import unittest
import numpy as np
from hgb_model import (
    train_test_split_stratified,
    CustomHistGradientBoostingClassifier,
    HistBinMapper,
    compute_f1_score,
    compute_roc_auc,
)


class TestLeakage(unittest.TestCase):

    def test_no_overlap_train_test(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 10)
        y = rng.randint(0, 2, size=1000)

        X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split_stratified(
            X, y, test_size=0.2, random_state=42, return_indices=True
        )
        # Check 1: Zero index intersection
        self.assertEqual(len(np.intersect1d(tr_idx, te_idx)), 0)
        self.assertEqual(len(tr_idx) + len(te_idx), len(y))

    def test_binning_leakage_isolation(self):
        # Verify bin mapper is fit ONLY on training data
        rng = np.random.RandomState(42)
        X_tr = rng.normal(0, 1, size=(500, 2))
        X_te = rng.normal(10, 1, size=(100, 2))

        mapper = HistBinMapper(max_bins=32)
        mapper.fit(X_tr)

        # Thresholds should reflect X_tr distribution (near [-2, 2]), not X_te
        for thr in mapper.bin_thresholds_:
            self.assertLess(np.max(thr), 6.0)

        # Transform on unseen X_te succeeds using X_tr thresholds
        X_te_b = mapper.transform(X_te)
        self.assertEqual(X_te_b.shape, X_te.shape)

    def test_two_phase_pipeline(self):
        # Phase 1: Dev model with validation
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 8).astype(np.float32)
        y = (X[:, 0] * 1.5 - X[:, 1] > 0).astype(np.float32)

        X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y, test_size=0.2, random_state=42)

        dev_model = CustomHistGradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
        dev_model.fit(X_tr, y_tr)
        best_iter = dev_model.best_n_iter_
        self.assertGreater(best_iter, 0)
        self.assertIsNotNone(dev_model.X_val_)

        # Threshold chosen ONLY on dev_model.X_val_
        p_val = dev_model.predict_proba(dev_model.X_val_)
        best_th = 0.5
        best_f1 = -1.0
        for th in [0.3, 0.4, 0.5, 0.6]:
            f1 = compute_f1_score(dev_model.y_val_, (p_val >= th).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        # Phase 2: Final model trained on FULL X_tr (100% of 800 samples)
        final_model = CustomHistGradientBoostingClassifier(
            n_estimators=best_iter,
            learning_rate=0.1,
            validation_fraction=0.0,
            n_iter_no_change=0,
            random_state=42
        )
        final_model.fit(X_tr, y_tr)

        # Phase 3: Final evaluation on untouched X_te
        y_test_pred = final_model.predict(X_te, threshold=best_th)
        y_test_proba = final_model.predict_proba(X_te)
        test_auc = compute_roc_auc(y_te, y_test_proba)
        self.assertGreaterEqual(test_auc, 0.0)
        self.assertLessEqual(test_auc, 1.0)
        self.assertEqual(len(y_test_pred), len(X_te))


if __name__ == '__main__':
    unittest.main()
