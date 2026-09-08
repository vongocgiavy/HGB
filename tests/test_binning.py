"""
Unit tests for HistBinMapper in hgb_model (Standard unittest).
"""
import unittest
import numpy as np
from hgb_model import HistBinMapper


class TestBinning(unittest.TestCase):

    def test_binning_fit_transform(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)

        mapper = HistBinMapper(max_bins=64)
        mapper.fit(X)
        X_b = mapper.transform(X)

        self.assertEqual(X_b.shape, X.shape)
        self.assertEqual(X_b.dtype, np.uint8)
        self.assertLessEqual(X_b.max(), 64)
        self.assertGreaterEqual(X_b.min(), 0)

    def test_binning_feature_count_mismatch(self):
        mapper = HistBinMapper(max_bins=32)
        mapper.fit(np.ones((50, 4)))

        with self.assertRaises(ValueError):
            mapper.transform(np.ones((50, 3)))

    def test_binning_nan_inf_detection(self):
        mapper = HistBinMapper(max_bins=32)
        X_nan = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with self.assertRaises(ValueError):
            mapper.fit(X_nan)

        X_good = np.array([[1.0, 2.0], [3.0, 4.0]])
        mapper.fit(X_good)

        X_inf = np.array([[1.0, np.inf]])
        with self.assertRaises(ValueError):
            mapper.transform(X_inf)

    def test_binning_invalid_max_bins(self):
        with self.assertRaises(ValueError):
            HistBinMapper(max_bins=1)

        with self.assertRaises(ValueError):
            HistBinMapper(max_bins=300)


if __name__ == '__main__':
    unittest.main()
