"""
Unit tests cho các trường hợp nâng cao và trường hợp biên (Edge Cases & Advanced Features).
Bao gồm:
1. Kiểm tra tính nhất quán của Alias tham số (l2_reg, min_gain, lr, patience) trong set_params/get_params.
2. Kiểm tra CustomGridSearchCV với l2_regularization và min_gain_to_split.
3. Kiểm tra trường hợp biên đặc trưng hằng số (Constant Feature - không đổi).
4. Kiểm tra trường hợp biên max_bins=2 (Rời rạc hóa tối thiểu).
5. Kiểm tra tính hữu hạn và giới hạn xác suất predict_proba (không bao giờ trả về đúng 0.0 hoặc 1.0).
6. Kiểm tra an toàn khi tất cả mẫu thuộc cùng một lớp (Homogeneous class leaf).
"""
import unittest
import numpy as np

from hgb_model import (
    CustomHistGradientBoostingClassifier,
    HistBinMapper,
    CustomGridSearchCV,
    compute_roc_auc,
)


class TestAdvancedAndEdgeCases(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Sinh dữ liệu đồ chơi 200 mẫu x 4 đặc trưng
        self.X = np.random.randn(200, 4).astype(np.float32)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.float32)

    def test_param_alias_consistency(self):
        """Kiểm tra set_params với các alias (l2_reg, min_gain, lr, patience) luôn cập nhật đồng nhất."""
        clf = CustomHistGradientBoostingClassifier()

        # 1. Alias l2_reg
        clf.set_params(l2_reg=5.5)
        self.assertEqual(clf.l2_reg, 5.5)
        self.assertEqual(clf.l2_regularization, 5.5)

        # 2. Alias min_gain
        clf.set_params(min_gain=0.05)
        self.assertEqual(clf.min_gain, 0.05)
        self.assertEqual(clf.min_gain_to_split, 0.05)

        # 3. Alias lr
        clf.set_params(lr=0.08)
        self.assertEqual(clf.learning_rate, 0.08)
        self.assertEqual(clf.lr, 0.08)

        # 4. Alias patience
        clf.set_params(patience=15)
        self.assertEqual(clf.n_iter_no_change, 15)
        self.assertEqual(clf.patience, 15)

        # Kiểm tra get_params trả về canonical keys
        params = clf.get_params()
        self.assertEqual(params['l2_regularization'], 5.5)
        self.assertEqual(params['min_gain_to_split'], 0.05)
        self.assertEqual(params['learning_rate'], 0.08)
        self.assertEqual(params['n_iter_no_change'], 15)

    def test_grid_search_with_l2_and_aliases(self):
        """Kiểm tra CustomGridSearchCV quét cả l2_regularization và min_gain_to_split."""
        base_clf = CustomHistGradientBoostingClassifier(
            n_estimators=10, max_depth=3, min_samples_leaf=10, random_state=42, validation_fraction=0.0
        )
        param_grid = {
            'l2_regularization': [0.1, 5.0],
            'min_gain_to_split': [1e-4, 1e-2],
        }
        gs = CustomGridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=2,
            scoring='roc_auc',
            verbose=0
        )
        gs.fit(self.X, self.y)

        self.assertIn('l2_regularization', gs.best_params_)
        self.assertIn('min_gain_to_split', gs.best_params_)
        self.assertIsNotNone(gs.best_estimator_)
        self.assertEqual(gs.best_estimator_.l2_regularization, gs.best_params_['l2_regularization'])
        self.assertEqual(gs.best_estimator_.l2_reg, gs.best_params_['l2_regularization'])

    def test_constant_feature_handling(self):
        """Kiểm tra khi có một đặc trưng là hằng số (variance=0), mô hình không bị lỗi chia cho 0."""
        X_const = self.X.copy()
        X_const[:, 1] = 7.0  # Cột 1 không đổi

        mapper = HistBinMapper(max_bins=255)
        mapper.fit(X_const)
        X_binned = mapper.transform(X_const)
        self.assertIn(mapper.n_bins_per_feature_[1], [1, 2])
        self.assertEqual(len(np.unique(X_binned[:, 1])), 1)  # Toàn bộ mẫu rơi vào cùng 1 bin

        clf = CustomHistGradientBoostingClassifier(
            n_estimators=10, max_depth=4, validation_fraction=0.1, random_state=42
        )
        clf.fit(X_const, self.y)

        preds = clf.predict(X_const)
        probs = clf.predict_proba(X_const)
        self.assertEqual(len(preds), len(self.y))
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())

    def test_binary_max_bins_2(self):
        """Kiểm tra trường hợp biên max_bins=2."""
        clf = CustomHistGradientBoostingClassifier(
            n_estimators=10, max_bins=2, max_depth=3, validation_fraction=0.0, random_state=42
        )
        clf.fit(self.X, self.y)
        probs = clf.predict_proba(self.X)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_predict_proba_bounds_clipping(self):
        """Kiểm tra predict_proba luôn nằm nghiêm ngặt trong (0.0, 1.0) do Sigmoid clip tại [-15, 15]."""
        clf = CustomHistGradientBoostingClassifier()

        # Logits cực đại và cực tiểu
        extreme_logits = np.array([-1000.0, -15.0, 0.0, 15.0, 1000.0], dtype=np.float32)
        sig_probs = clf._sigmoid(extreme_logits)

        # Xác suất không bao giờ là đúng 0.0 hoặc 1.0
        self.assertTrue(np.all(sig_probs > 0.0))
        self.assertTrue(np.all(sig_probs < 1.0))
        self.assertAlmostEqual(sig_probs[0], 1.0 / (1.0 + np.exp(15.0)), places=6)
        self.assertAlmostEqual(sig_probs[-1], 1.0 / (1.0 + np.exp(-15.0)), places=6)

    def test_single_class_validation_and_imbalance_leaf_safety(self):
        """Kiểm tra: (1) ném lỗi khi chỉ có 1 nhãn duy nhất; (2) huấn luyện an toàn khi mất cân bằng cực đoan (tạo lá thuần nhất)."""
        # 1. Toàn bộ mẫu cùng nhãn -> Bắt buộc ném ValueError
        y_all_ones = np.ones(len(self.X), dtype=np.float32)
        clf = CustomHistGradientBoostingClassifier(n_estimators=5, random_state=42)
        with self.assertRaises(ValueError):
            clf.fit(self.X, y_all_ones)

        # 2. Mất cân bằng cực đoan (1 mẫu lớp 0, 199 mẫu lớp 1 -> các lá nhanh chóng thuần nhất)
        y_imbalanced = np.ones(len(self.X), dtype=np.float32)
        y_imbalanced[0] = 0.0
        clf_imb = CustomHistGradientBoostingClassifier(
            n_estimators=5, max_depth=4, validation_fraction=0.0, random_state=42
        )
        clf_imb.fit(self.X, y_imbalanced)
        probs = clf_imb.predict_proba(self.X)
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())


if __name__ == "__main__":
    unittest.main()
