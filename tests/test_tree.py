"""
Unit tests for HistTreeNode and HistRegressionTree in hgb_model (Standard unittest).
"""
import unittest
import numpy as np
from hgb_model import HistRegressionTree, HistBinMapper


class TestTree(unittest.TestCase):

    def test_tree_construction_and_predict(self):
        rng = np.random.RandomState(42)
        n_samples, n_features = 300, 4
        X = rng.randn(n_samples, n_features).astype(np.float32)

        mapper = HistBinMapper(max_bins=64)
        mapper.fit(X)
        X_b = mapper.transform(X)

        g = (rng.rand(n_samples) - 0.5).astype(np.float32)
        h = np.ones(n_samples, dtype=np.float32)

        tree = HistRegressionTree(max_depth=4, min_samples_leaf=15, l2_regularization=1.0, max_bins=64)
        tree.fit(X_b, g, h)

        preds = tree.predict(X_b)
        self.assertEqual(preds.shape, (n_samples,))
        self.assertTrue(np.isfinite(preds).all())

        importances = tree.compute_feature_importances(n_features)
        self.assertEqual(len(importances), n_features)
        self.assertTrue(np.all(importances >= 0.0))

    def test_tree_depth_and_min_leaf(self):
        rng = np.random.RandomState(42)
        X_b = rng.randint(0, 32, size=(100, 3), dtype=np.uint8)
        g = rng.randn(100).astype(np.float32)
        h = np.ones(100, dtype=np.float32)

        tree = HistRegressionTree(max_depth=2, min_samples_leaf=10)
        tree.fit(X_b, g, h)

        def get_max_depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(get_max_depth(node.left), get_max_depth(node.right))

        actual_depth = get_max_depth(tree.root)
        self.assertLessEqual(actual_depth, 2)


if __name__ == '__main__':
    unittest.main()
