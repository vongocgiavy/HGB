# -*- coding: utf-8 -*-
"""
Unit tests for weights.py (Model Weights Serialization, Deserialization,
FastInferenceEngine, and Weights Folder Structure).
"""
import os
import unittest
import tempfile
import numpy as np

from hgb_model import CustomHistGradientBoostingClassifier
from weights import (
    export_weights_to_json,
    export_weights_to_npz,
    export_weights_to_txt,
    load_model_from_weights,
    FastInferenceEngine,
    FEATURE_NAMES,
)


class TestWeights(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(200, 5).astype(np.float32)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(np.int32)
        self.feature_names = [f"f_{i}" for i in range(5)]

        self.clf = CustomHistGradientBoostingClassifier(
            n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.clf.fit(self.X, self.y)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_and_load_roundtrip_json(self):
        json_path = os.path.join(self.temp_dir.name, "test_weights.json")
        export_weights_to_json(
            self.clf, json_path,
            feature_names=self.feature_names,
            optimal_threshold=0.45,
            metrics={"accuracy": 0.85, "roc_auc": 0.90}
        )
        self.assertTrue(os.path.exists(json_path))

        # Khôi phục mô hình từ file json
        reconstructed = load_model_from_weights(json_path)
        self.assertEqual(len(reconstructed.trees), len(self.clf.trees))
        self.assertEqual(reconstructed.n_features_in_, self.clf.n_features_in_)

        # So sánh xác suất dự đoán (parity)
        p_orig = self.clf.predict_proba(self.X)
        p_recon = reconstructed.predict_proba(self.X)
        np.testing.assert_allclose(p_orig, p_recon, atol=1e-5)

    def test_export_npz(self):
        npz_path = os.path.join(self.temp_dir.name, "test_weights.npz")
        export_weights_to_npz(
            self.clf, npz_path,
            feature_names=self.feature_names,
            optimal_threshold=0.45,
            metrics={"accuracy": 0.85, "roc_auc": 0.90}
        )
        self.assertTrue(os.path.exists(npz_path))

        data = np.load(npz_path)
        self.assertIn("base_score", data)
        self.assertIn("tree_ids", data)
        self.assertIn("leaf_weights", data)
        self.assertIn("bin_thresholds_flat", data)
        self.assertEqual(int(data["n_trees"]), len(self.clf.trees))

    def test_export_txt(self):
        txt_path = os.path.join(self.temp_dir.name, "test_weights.txt")
        export_weights_to_txt(
            self.clf, txt_path,
            feature_names=self.feature_names,
            optimal_threshold=0.45,
            metrics={"accuracy": 0.85, "roc_auc": 0.90}
        )
        self.assertTrue(os.path.exists(txt_path))
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("BÁO CÁO TOÀN DIỆN TRỌNG SỐ MÔ HÌNH", content)
        self.assertIn("THÔNG TIN SIÊU THAM SỐ", content)
        self.assertIn("Accuracy", content)

    def test_fast_inference_engine(self):
        json_path = os.path.join(self.temp_dir.name, "test_weights.json")
        export_weights_to_json(
            self.clf, json_path,
            feature_names=self.feature_names,
            optimal_threshold=0.45
        )
        engine = FastInferenceEngine(json_path)
        self.assertEqual(engine.threshold, 0.45)

        p_orig = self.clf.predict_proba(self.X)
        p_engine = engine.predict_proba(self.X)
        np.testing.assert_allclose(p_orig, p_engine, atol=1e-5)

        preds_engine = engine.predict(self.X)
        expected_preds = (p_orig >= 0.45).astype(int)
        np.testing.assert_array_equal(preds_engine, expected_preds)

    def test_weights_folder_structure(self):
        # Kiểm tra sự tồn tại của 4 file chính xác như ảnh yêu cầu
        hgb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(hgb_dir, "weights")

        expected_files = [
            "baseline_weights.json",
            "baseline_weights.npz",
            "baseline_weights.txt",
            "best_model_weights.json",
        ]

        for fname in expected_files:
            fpath = os.path.join(weights_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Thiếu file trọng số: {fname}")
            self.assertGreater(os.path.getsize(fpath), 0, f"File rỗng: {fname}")


if __name__ == '__main__':
    unittest.main()
