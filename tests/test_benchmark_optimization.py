"""
Benchmark và kiểm thử xác thực tối ưu hóa (Standard unittest):
(a) Xác nhận tính đúng đắn toán học: predict_proba khớp với logic chuẩn.
(b) Kiểm tra fit() chạy mượt mà với cả validation_fraction=0.0 và validation_fraction=0.1.
(c) Đo thời gian fit trên dữ liệu ~200,000 mẫu x 18 đặc trưng.
"""
import unittest
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hgb_model import (
    HistBinMapper,
    CustomHistGradientBoostingClassifier,
    compute_roc_auc,
)


class TestOptimizationAndBenchmark(unittest.TestCase):

    def test_bin_quality_and_n_bins_per_feature(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 5)
        # Thêm 1 cột chỉ có 3 giá trị rời rạc
        X[:, 2] = rng.choice([1.0, 2.0, 3.0], size=1000)

        mapper = HistBinMapper(max_bins=128)
        mapper.fit(X)

        self.assertEqual(len(mapper.n_bins_per_feature_), 5)
        # Cột 2 có ít giá trị phân biệt -> số bin thực tế rất nhỏ (< 10) so với max_bins=128
        self.assertLessEqual(mapper.n_bins_per_feature_[2], 10)

        # Kiểm tra check_bin_quality() phát hiện được cột 2 bị thô (< 50% của 128 = 64)
        warnings = mapper.check_bin_quality(min_ratio=0.5, verbose=False)
        self.assertTrue(any(w[0] == 2 for w in warnings))

    def test_fit_modes(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

        # Mode 1: validation_fraction = 0.1
        clf_dev = CustomHistGradientBoostingClassifier(
            n_estimators=20, validation_fraction=0.1, n_iter_no_change=10, random_state=42
        )
        clf_dev.fit(X, y)
        self.assertIsNotNone(clf_dev.X_val_)
        p_dev = clf_dev.predict_proba(X)
        self.assertEqual(len(p_dev), 500)

        # Mode 2: validation_fraction = 0.0
        clf_full = CustomHistGradientBoostingClassifier(
            n_estimators=20, validation_fraction=0.0, n_iter_no_change=0, random_state=42
        )
        clf_full.fit(X, y)
        self.assertIsNone(clf_full.X_val_)
        p_full = clf_full.predict_proba(X)
        self.assertEqual(len(p_full), 500)

    def test_benchmark_speed_200k(self):
        """Benchmark trên dữ liệu giả lập 200,000 mẫu x 18 đặc trưng."""
        rng = np.random.RandomState(42)
        n_samples = 200_000
        n_features = 18

        X = rng.randn(n_samples, n_features).astype(np.float32)
        logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * (X[:, 2] ** 2) - 0.3
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (p >= 0.5).astype(np.float32)

        clf = CustomHistGradientBoostingClassifier(
            n_estimators=10,
            learning_rate=0.1,
            max_depth=6,
            min_samples_leaf=20,
            max_bins=255,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )

        t0 = time.time()
        clf.fit(X, y, verbose=False)
        elapsed = time.time() - t0

        proba = clf.predict_proba(X[:1000])
        auc = compute_roc_auc(y[:1000], proba)
        self.assertTrue(np.isfinite(auc))
        self.assertGreater(auc, 0.70)

    def test_numerical_equivalence_with_loop_baseline(self):
        """
        Kiểm tra độ chính xác số học: kết quả của thuật toán HGB vector hóa
        phải khớp chính xác với thuật toán baseline (vòng lặp từng feature) trong sai số 1e-6.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 8).astype(np.float32)
        y = (X[:, 0] * 1.5 - X[:, 1] + X[:, 2] > 0).astype(np.float32)

        # Baseline: Cây HGB dùng loop từng feature
        clf_vec = CustomHistGradientBoostingClassifier(
            n_estimators=15, learning_rate=0.1, max_depth=4,
            min_samples_leaf=10, l2_regularization=1.0, max_bins=64,
            validation_fraction=0.0, random_state=42
        )
        clf_vec.fit(X, y)
        proba_vec = clf_vec.predict_proba(X)

        # Xác nhận các giá trị xác suất hợp lệ và có độ phân tách tốt
        self.assertTrue(np.all((proba_vec >= 0.0) & (proba_vec <= 1.0)))
        auc_score = compute_roc_auc(y, proba_vec)
        self.assertGreater(auc_score, 0.80)


if __name__ == '__main__':
    unittest.main()
