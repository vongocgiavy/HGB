# -*- coding: utf-8 -*-
"""
Module Quản Lý Trọng Số Mô Hình (Model Weights Management)
==========================================================
Dự án: Histogram Gradient Boosting (HGB) - Phân loại va chạm hạt SUSY
Triển khai: 100% Thuần Python & NumPy (Zero Scikit-Learn)

Cung cấp các công cụ:
- Trích xuất và lưu toàn bộ trọng số ra các định dạng chuẩn:
    + JSON (.json): Cấu trúc phân cấp đầy đủ, metadata, metrics, ngưỡng bin và cây quyết định.
    + NumPy Archive (.npz): Lưu trữ nhị phân tốc độ cao, zero-copy, tối ưu bộ nhớ.
    + Báo cáo Text (.txt): Báo cáo trực quan, học thuật, dễ đọc cho con người.
- Khôi phục mô hình hoàn chỉnh từ trọng số mà không cần dữ liệu huấn luyện.
- FastInferenceEngine: Bộ suy luận độc lập siêu tốc trực tiếp từ trọng số.
- CLI thực thi một lần duy nhất (không chạy lặp liên tục) để sinh đầy đủ thư mục weights:
    weights/
    ├── baseline_weights.json
    ├── baseline_weights.npz
    ├── baseline_weights.txt
    └── best_model_weights.json
"""

import os
import sys
import json
import time
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import các lớp và hàm đánh giá từ hgb_model
from hgb_model import (
    HistTreeNode,
    HistRegressionTree,
    HistBinMapper,
    CustomHistGradientBoostingClassifier,
    train_test_split_stratified,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_specificity,
    compute_npv,
    compute_f1_score,
    compute_roc_auc,
    compute_confusion_matrix,
)

FEATURE_NAMES = [
    "lepton1_pT", "lepton1_eta", "lepton1_phi",
    "lepton2_pT", "lepton2_eta", "lepton2_phi",
    "MET_magnitude", "MET_phi",
    "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2", "S_R",
    "M_Delta_R", "dPhi_r_b", "cos_theta_r1"
]

FEATURE_DESCRIPTIONS = {
    "lepton1_pT":    "Transverse momentum lepton 1 [Low-level]",
    "lepton1_eta":   "Pseudorapidity lepton 1 [Low-level]",
    "lepton1_phi":   "Azimuthal angle lepton 1 [Low-level]",
    "lepton2_pT":    "Transverse momentum lepton 2 [Low-level]",
    "lepton2_eta":   "Pseudorapidity lepton 2 [Low-level]",
    "lepton2_phi":   "Azimuthal angle lepton 2 [Low-level]",
    "MET_magnitude": "Missing Transverse Energy magnitude [Low-level]",
    "MET_phi":       "MET azimuthal angle [Low-level]",
    "MET_rel":       "MET relative to nearest jet [High-level]",
    "axial_MET":     "Axial Missing ET [High-level]",
    "M_R":           "Razor mass M_R [High-level]",
    "M_TR_2":        "Transverse razor mass M_TR_2 [High-level]",
    "R":             "Razor ratio R [High-level]",
    "MT2":           "Stransverse mass MT2 [High-level]",
    "S_R":           "Super-razor variable S_R [High-level]",
    "M_Delta_R":     "Super-razor M_Delta_R [High-level]",
    "dPhi_r_b":      "Azimuthal angle dPhi_r_b [High-level]",
    "cos_theta_r1":  "cos(theta_r1) Razor frame decay angle [High-level]",
}


# ==============================================================================
# HÀM CHUYỂN ĐỔI CÂY (TREE SERIALIZATION HELPERS)
# ==============================================================================

def _node_to_dict(node, feature_names=None):
    """Chuyển đổi đệ quy một nút cây HistTreeNode sang dictionary."""
    if node is None:
        return None
    if node.is_leaf:
        return {
            "type": "leaf",
            "is_leaf": True,
            "weight": round(float(node.value), 6)
        }
    
    feat_name = feature_names[node.feature_idx] if (feature_names and node.feature_idx < len(feature_names)) else None
    return {
        "type": "split",
        "is_leaf": False,
        "feature_index": int(node.feature_idx),
        "feature_name": feat_name,
        "bin_threshold": int(node.bin_threshold),
        "gain": round(float(node.gain), 6),
        "left_child": _node_to_dict(node.left, feature_names),
        "right_child": _node_to_dict(node.right, feature_names)
    }


def _dict_to_node(d):
    """Khôi phục đệ quy nút cây HistTreeNode từ dictionary."""
    if d is None:
        return None
    if d.get("is_leaf", False) or d.get("type") == "leaf":
        val = d.get("weight", d.get("value", 0.0))
        return HistTreeNode(is_leaf=True, value=float(val))
    
    node = HistTreeNode(
        is_leaf=False,
        feature_idx=d.get("feature_index", d.get("feature_idx")),
        bin_threshold=d.get("bin_threshold"),
        gain=d.get("gain", 0.0)
    )
    node.left = _dict_to_node(d.get("left_child", d.get("left")))
    node.right = _dict_to_node(d.get("right_child", d.get("right")))
    return node


def _flatten_tree(root, feature_names=None):
    """Biến đổi cây thành danh sách nút phẳng phục vụ xuất mảng NumPy (.npz)."""
    nodes = []
    if root is None:
        return nodes

    # Duyệt theo thứ tự cấp độ (Level-order / BFS) để có node ID nhất quán
    queue = [root]
    node_to_id = {}
    curr_id = 0

    while queue:
        n = queue.pop(0)
        node_to_id[id(n)] = curr_id
        curr_id += 1
        if not n.is_leaf:
            if n.left is not None:
                queue.append(n.left)
            if n.right is not None:
                queue.append(n.right)

    # Duyệt lần 2 để tạo danh sách bản ghi
    queue = [root]
    while queue:
        n = queue.pop(0)
        nid = node_to_id[id(n)]
        left_id = node_to_id[id(n.left)] if (not n.is_leaf and n.left is not None) else -1
        right_id = node_to_id[id(n.right)] if (not n.is_leaf and n.right is not None) else -1

        nodes.append({
            "node_id": nid,
            "is_leaf": bool(n.is_leaf),
            "weight": float(n.value) if n.is_leaf else 0.0,
            "feature_idx": int(n.feature_idx) if not n.is_leaf else -1,
            "bin_threshold": int(n.bin_threshold) if not n.is_leaf else -1,
            "gain": float(n.gain) if not n.is_leaf else 0.0,
            "left_child": left_id,
            "right_child": right_id,
        })
        if not n.is_leaf:
            if n.left is not None:
                queue.append(n.left)
            if n.right is not None:
                queue.append(n.right)

    return nodes


# ==============================================================================
# HÀM XUẤT TRỌNG SỐ (EXPORT FUNCTIONS)
# ==============================================================================

def export_weights_to_json(model: CustomHistGradientBoostingClassifier, filepath: str,
                           feature_names=None, optimal_threshold: float = 0.5,
                           metrics: dict = None, model_type: str = "model"):
    """
    Trích xuất toàn bộ trọng số của mô hình HGB ra tệp JSON cấu trúc rõ ràng.
    """
    model._check_is_fitted()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    # 1. Trích xuất bin thresholds từ HistBinMapper
    bin_thresholds_list = []
    if model.bin_mapper and hasattr(model.bin_mapper, 'bin_thresholds_'):
        for th in model.bin_mapper.bin_thresholds_:
            bin_thresholds_list.append([round(float(v), 6) for v in th])

    # 2. Trích xuất toàn bộ cây hồi quy
    trees_list = []
    for i, tree in enumerate(model.trees):
        trees_list.append({
            "tree_id": i + 1,
            "max_depth": tree.max_depth,
            "root": _node_to_dict(tree.root, feature_names)
        })

    # 3. Đóng gói payload trọng số
    weights_payload = {
        "metadata": {
            "model_architecture": "Histogram Gradient Boosting Classifier",
            "version": "1.0.0",
            "model_type": model_type,
            "implementation": "100% Native Python & NumPy (Zero Scikit-Learn)",
            "n_estimators": len(model.trees),
            "learning_rate": float(model.learning_rate),
            "base_score_log_odds": round(float(model.base_score_), 6),
            "base_probability": round(float(1.0 / (1.0 + np.exp(-model.base_score_))), 6),
            "optimal_classification_threshold": round(float(optimal_threshold), 4),
            "n_features": int(model.n_features_in_),
            "feature_names": feature_names,
            "metrics": metrics or {},
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
        "feature_importances": {
            feature_names[i]: round(float(model.feature_importances_[i]), 6)
            for i in range(model.n_features_in_)
        } if hasattr(model, 'feature_importances_') else {},
        "quantization_bins": {
            "max_bins": getattr(model.bin_mapper, 'max_bins', 255),
            "feature_thresholds": {
                feature_names[i]: bin_thresholds_list[i] if i < len(bin_thresholds_list) else []
                for i in range(len(feature_names))
            }
        },
        "trees": trees_list
    }

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(weights_payload, f, indent=2, ensure_ascii=False)

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"  [✓] Đã xuất JSON trọng số: {filepath} ({file_size_kb:.1f} KB, {len(trees_list)} cây)")
    return filepath


def export_weights_to_npz(model: CustomHistGradientBoostingClassifier, filepath: str,
                          feature_names=None, optimal_threshold: float = 0.5,
                          metrics: dict = None):
    """
    Trích xuất toàn bộ trọng số mô hình thành tệp lưu trữ nén nhị phân NumPy (.npz).
    Cho phép nạp lại cực nhanh (zero-copy) trực tiếp bằng np.load().
    """
    model._check_is_fitted()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    # 1. Thu thập mảng bin thresholds
    all_thresholds = []
    threshold_splits = [0]
    for th in model.bin_mapper.bin_thresholds_:
        all_thresholds.extend(th)
        threshold_splits.append(len(all_thresholds))

    bin_thresholds_flat = np.array(all_thresholds, dtype=np.float32)
    bin_thresholds_splits = np.array(threshold_splits, dtype=np.int32)

    # 2. Thu thập mảng cây quyết định
    tree_ids = []
    node_ids = []
    is_leaf = []
    leaf_weights = []
    feature_indices = []
    bin_thresholds = []
    gains = []
    left_children = []
    right_children = []

    for t_idx, tree in enumerate(model.trees):
        tree_nodes = _flatten_tree(tree.root, feature_names)
        for nd in tree_nodes:
            tree_ids.append(t_idx + 1)
            node_ids.append(nd["node_id"])
            is_leaf.append(nd["is_leaf"])
            leaf_weights.append(nd["weight"])
            feature_indices.append(nd["feature_idx"])
            bin_thresholds.append(nd["bin_threshold"])
            gains.append(nd["gain"])
            left_children.append(nd["left_child"])
            right_children.append(nd["right_child"])

    # 3. Thu thập metrics
    metrics_keys = ["accuracy", "precision", "recall", "specificity", "npv", "f1_score", "roc_auc"]
    metrics_vals = [float(metrics.get(k, 0.0)) if metrics else 0.0 for k in metrics_keys]

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(
        filepath,
        # Metadata
        base_score=np.float32(model.base_score_),
        learning_rate=np.float32(model.learning_rate),
        optimal_threshold=np.float32(optimal_threshold),
        n_features=np.int32(model.n_features_in_),
        max_bins=np.int32(getattr(model.bin_mapper, 'max_bins', 255)),
        n_trees=np.int32(len(model.trees)),
        feature_names=np.array(feature_names, dtype=str),
        feature_importances=np.array(model.feature_importances_, dtype=np.float64),
        # Bin mapper thresholds
        bin_thresholds_flat=bin_thresholds_flat,
        bin_thresholds_splits=bin_thresholds_splits,
        # Tree structures & Leaf weights
        tree_ids=np.array(tree_ids, dtype=np.int32),
        node_ids=np.array(node_ids, dtype=np.int32),
        is_leaf=np.array(is_leaf, dtype=bool),
        leaf_weights=np.array(leaf_weights, dtype=np.float32),
        feature_indices=np.array(feature_indices, dtype=np.int16),
        bin_thresholds=np.array(bin_thresholds, dtype=np.int16),
        gains=np.array(gains, dtype=np.float32),
        left_children=np.array(left_children, dtype=np.int32),
        right_children=np.array(right_children, dtype=np.int32),
        # Metrics
        metrics_keys=np.array(metrics_keys, dtype=str),
        metrics_values=np.array(metrics_vals, dtype=np.float32)
    )

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"  [✓] Đã xuất NPZ trọng số: {filepath} ({file_size_kb:.1f} KB)")
    return filepath


def export_weights_to_txt(model: CustomHistGradientBoostingClassifier, filepath: str,
                          feature_names=None, optimal_threshold: float = 0.5,
                          metrics: dict = None, model_type: str = "baseline"):
    """
    Xuất báo cáo trọng số trực quan, chi tiết dạng văn bản (.txt).
    Bao gồm cấu hình siêu tham số, phân bổ trọng số lá, tầm quan trọng đặc trưng,
    bảng lượng tử hóa bin và cấu trúc các cây tiêu biểu.
    """
    model._check_is_fitted()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    width = 86
    sep_double = "=" * width
    sep_single = "-" * width

    lines = [
        sep_double,
        f"   BÁO CÁO TOÀN DIỆN TRỌNG SỐ MÔ HÌNH HGB ({model_type.upper()}_WEIGHTS)",
        "   Dự án: Phân loại biến cố va chạm hạt SUSY (UCI Benchmark)",
        "   Triển khai: 100% Python thuần và NumPy (Zero Scikit-Learn)",
        sep_double,
        "",
        "1. THÔNG TIN SIÊU THAM SỐ & ĐỘ PHÂN GIẢI",
        sep_single,
        f"  • Loại mô hình             : {model_type}",
        f"  • Số cây quyết định        : {len(model.trees)} cây (n_estimators={len(model.trees)})",
        f"  • Tốc độ học (shrinkage)   : {model.learning_rate}",
        f"  • Hệ số chính quy L2       : {getattr(model, 'l2_reg', 1.0)}",
        f"  • Độ sâu tối đa mỗi cây    : {getattr(model.trees[0], 'max_depth', 6) if model.trees else 'N/A'}",
        f"  • Số mẫu tối thiểu nút lá  : {getattr(model.trees[0], 'min_samples_leaf', 20) if model.trees else 'N/A'}",
        f"  • Số bin lượng tử hóa max  : {getattr(model.bin_mapper, 'max_bins', 255)}",
        f"  • Điểm khởi tạo log-odds   : {model.base_score_:.6f}",
        f"  • Xác suất tiên nghiệm     : {1.0 / (1.0 + np.exp(-model.base_score_)):.4f}",
        f"  • Ngưỡng phân loại tối ưu  : tau* = {optimal_threshold:.2f}",
        f"  • Thời điểm xuất file      : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        "",
    ]

    # Metrics nếu có
    if metrics:
        lines.extend([
            "2. CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ (TEST SET)",
            sep_single,
            f"  • Accuracy (Độ chính xác)     : {metrics.get('accuracy', 0.0)*100:6.2f}%",
            f"  • Precision (Độ chuẩn xác)    : {metrics.get('precision', 0.0)*100:6.2f}%",
            f"  • Recall (Độ nhạy - TPR)      : {metrics.get('recall', 0.0)*100:6.2f}%",
            f"  • Specificity (Độ đặc hiệu)   : {metrics.get('specificity', 0.0)*100:6.2f}%",
            f"  • NPV (Giá trị dự đoán âm)    : {metrics.get('npv', 0.0)*100:6.2f}%",
            f"  • F1-Score                    : {metrics.get('f1_score', 0.0)*100:6.2f}%",
            f"  • ROC-AUC                     : {metrics.get('roc_auc', 0.0):6.4f}",
            "",
        ])

    # Bảng Feature Importances
    lines.extend([
        "3. BẢNG TẦM QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCES - GAIN)",
        sep_single,
        f"  {'Hạng':<5} | {'Tên đặc trưng':<18} | {'Tỷ lệ Gain':>10} | Mô tả đặc trưng",
        sep_single,
    ])
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    for r, idx in enumerate(sorted_idx, 1):
        fn = feature_names[idx]
        gain_pct = model.feature_importances_[idx] * 100
        desc = FEATURE_DESCRIPTIONS.get(fn, "")
        lines.append(f"  #{r:<4} | {fn:<18} | {gain_pct:9.2f}% | {desc}")
    lines.append("")

    # Bảng Lượng tử hóa Bin
    lines.extend([
        "4. BẢNG LƯỢNG TỬ HÓA BIN (QUANTIZATION BINS PER FEATURE)",
        sep_single,
        f"  {'#':<3} | {'Đặc trưng':<18} | {'Số bin':>8} | {'Ngưỡng Min':>12} | {'Ngưỡng Max':>12}",
        sep_single,
    ])
    for j, fn in enumerate(feature_names):
        ths = model.bin_mapper.bin_thresholds_[j] if j < len(model.bin_mapper.bin_thresholds_) else []
        n_b = len(ths) + 1 if len(ths) > 0 else 1
        t_min = f"{ths[0]:.4f}" if len(ths) > 0 else "N/A"
        t_max = f"{ths[-1]:.4f}" if len(ths) > 0 else "N/A"
        lines.append(f"  {j+1:<3} | {fn:<18} | {n_b:8d} | {t_min:>12} | {t_max:>12}")
    lines.append("")

    # Phân tích thống kê các nút lá trên toàn bộ ensemble
    all_leaf_weights = []
    total_nodes = 0
    total_leaves = 0
    for tree in model.trees:
        nodes = _flatten_tree(tree.root, feature_names)
        total_nodes += len(nodes)
        for nd in nodes:
            if nd["is_leaf"]:
                total_leaves += 1
                all_leaf_weights.append(nd["weight"])

    all_leaf_weights = np.array(all_leaf_weights, dtype=np.float32)
    lines.extend([
        "5. THỐNG KÊ PHÂN BỔ TRỌNG SỐ LÁ (LEAF WEIGHTS DISTRIBUTION)",
        sep_single,
        f"  • Tổng số nút trong rừng cây : {total_nodes:,} nút",
        f"  • Tổng số nút lá (leaves)    : {total_leaves:,} lá",
        f"  • Trọng số lá nhỏ nhất (Min) : {float(np.min(all_leaf_weights)):.6f}",
        f"  • Trọng số lá lớn nhất (Max) : {float(np.max(all_leaf_weights)):.6f}",
        f"  • Trọng số lá trung bình     : {float(np.mean(all_leaf_weights)):.6f}",
        f"  • Độ lệch chuẩn trọng số lá  : {float(np.std(all_leaf_weights)):.6f}",
        "",
    ])

    # Trực quan hóa cấu trúc Cây #1 và Cây #2
    def _tree_to_ascii(node, prefix="", is_left=True, depth=0, max_display_depth=3):
        if node is None or depth > max_display_depth:
            return []
        res = []
        pointer = "├── [L] " if is_left else "└── [R] "
        if depth == 0:
            pointer = "ROOT: "
        if node.is_leaf:
            res.append(f"{prefix}{pointer}LEAF weight = {node.value:+.5f}")
        else:
            fn = feature_names[node.feature_idx]
            res.append(f"{prefix}{pointer}SPLIT {fn} (bin <= {node.bin_threshold}) | gain={node.gain:.4f}")
            new_prefix = prefix + ("│   " if is_left else "    ") if depth > 0 else "  "
            res.extend(_tree_to_ascii(node.left, new_prefix, True, depth + 1, max_display_depth))
            res.extend(_tree_to_ascii(node.right, new_prefix, False, depth + 1, max_display_depth))
        return res

    lines.extend([
        "6. CẤU TRÚC MẪU CỦA CÁC CÂY QUYẾT ĐỊNH BAN ĐẦU",
        sep_single,
    ])
    for t_idx in range(min(2, len(model.trees))):
        lines.append(f"  --- CÂY QUYẾT ĐỊNH #{t_idx+1} ---")
        lines.extend(_tree_to_ascii(model.trees[t_idx].root, prefix="  ", depth=0, max_display_depth=3))
        lines.append("")

    lines.extend([
        sep_double,
        "   KẾT THÚC BÁO CÁO TRỌNG SỐ",
        sep_double,
    ])

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"  [✓] Đã xuất TXT trọng số:  {filepath} ({file_size_kb:.1f} KB)")
    return filepath


# ==============================================================================
# HÀM TÁI TẠO MÔ HÌNH TỪ TRỌNG SỐ (DESERIALIZATION)
# ==============================================================================

def load_model_from_weights(filepath: str) -> CustomHistGradientBoostingClassifier:
    """
    Tái tạo hoàn chỉnh một đối tượng CustomHistGradientBoostingClassifier từ tệp JSON trọng số.
    Không cần nạp lại dữ liệu huấn luyện.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data["metadata"]
    n_features = meta["n_features"]
    learning_rate = meta["learning_rate"]
    base_score = meta["base_score_log_odds"]
    trees_data = data["trees"]
    max_bins = data["quantization_bins"]["max_bins"]

    # Khởi tạo mô hình rỗng
    model = CustomHistGradientBoostingClassifier(
        n_estimators=len(trees_data),
        learning_rate=learning_rate,
        max_bins=max_bins
    )
    model.base_score_ = float(base_score)
    model.n_features_in_ = int(n_features)

    # Khôi phục HistBinMapper
    model.bin_mapper = HistBinMapper(max_bins=max_bins)
    feat_names = meta.get("feature_names", [])
    th_dict = data["quantization_bins"]["feature_thresholds"]

    bin_thresholds_ = []
    for fname in feat_names:
        arr = np.array(th_dict.get(fname, []), dtype=np.float32)
        bin_thresholds_.append(arr)
    model.bin_mapper.bin_thresholds_ = bin_thresholds_
    model.bin_mapper.n_features_in_ = n_features

    # Khôi phục toàn bộ cây
    model.trees = []
    for t_dict in trees_data:
        tree = HistRegressionTree(max_bins=max_bins, max_depth=t_dict.get("max_depth", 6))
        tree.root = _dict_to_node(t_dict["root"])
        model.trees.append(tree)

    # Khôi phục Feature Importances
    imp_dict = data.get("feature_importances", {})
    if imp_dict:
        model.feature_importances_ = np.array([imp_dict.get(fname, 0.0) for fname in feat_names], dtype=np.float64)
    else:
        model.feature_importances_ = np.zeros(n_features, dtype=np.float64)

    return model


class FastInferenceEngine:
    """
    Động cơ suy luận độc lập siêu tốc (Standalone Fast Inference Engine).
    Nạp trực tiếp từ file JSON trọng số và thực hiện suy luận thời gian thực
    với hiệu năng cao nhất (Zero Scikit-Learn).
    """
    def __init__(self, weights_filepath: str):
        with open(weights_filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.meta = self.data["metadata"]
        self.n_features = self.meta["n_features"]
        self.learning_rate = self.meta["learning_rate"]
        self.base_score = self.meta["base_score_log_odds"]
        self.threshold = self.meta.get("optimal_classification_threshold", 0.5)
        self.feature_names = self.meta.get("feature_names", [])

        # Tiền xử lý ngưỡng chia bin dạng NumPy array để transform siêu nhanh
        th_dict = self.data["quantization_bins"]["feature_thresholds"]
        self.bin_thresholds = [np.array(th_dict.get(fn, []), dtype=np.float32) for fn in self.feature_names]

        # Nạp các cây hồi quy
        self.trees = []
        for t_dict in self.data["trees"]:
            self.trees.append(_dict_to_node(t_dict["root"]))

    def _transform_bins(self, X: np.ndarray) -> np.ndarray:
        """Rời rạc hóa ma trận đầu vào thành uint8."""
        N = X.shape[0]
        X_b = np.empty((N, self.n_features), dtype=np.uint8)
        for j in range(self.n_features):
            th = self.bin_thresholds[j]
            if len(th) == 0:
                X_b[:, j] = 0
            else:
                X_b[:, j] = np.searchsorted(th, X[:, j], side='right').astype(np.uint8)
        return X_b

    def _predict_tree(self, root_node, X_b: np.ndarray) -> np.ndarray:
        """Duyệt ngăn xếp stack-based dự đoán 1 cây."""
        preds = np.zeros(X_b.shape[0], dtype=np.float32)
        if root_node is None:
            return preds
        stack = [(root_node, np.arange(len(preds), dtype=np.intp))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                preds[idx] = node.value
                continue
            mask = X_b[idx, node.feature_idx] <= node.bin_threshold
            left_idx = idx[mask]
            right_idx = idx[~mask]
            if len(right_idx) > 0:
                stack.append((node.right, right_idx))
            if len(left_idx) > 0:
                stack.append((node.left, left_idx))
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Tính xác suất dự đoán P(y=1|X)."""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_b = self._transform_bins(X)
        F = np.full(X_b.shape[0], self.base_score, dtype=np.float32)
        for root in self.trees:
            F += self.learning_rate * self._predict_tree(root, X_b)
        F_clipped = np.clip(F, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-F_clipped))

    def predict_proba_2d(self, X: np.ndarray) -> np.ndarray:
        """Trả về mảng 2 chiều [P(y=0|X), P(y=1|X)]."""
        p1 = self.predict_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Phân loại nhị phân theo ngưỡng chỉ định hoặc ngưỡng tối ưu lưu trong trọng số."""
        thr = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= thr).astype(int)


# ==============================================================================
# HÀM THỰC THI CHÍNH: HUẤN LUYỆN & XUẤT TRỌNG SỐ TOÀN BỘ (SINGLE RUN)
# ==============================================================================

def run_weights_pipeline():
    """
    Thực thi tạo bộ trọng số cho dự án:
    1. Huấn luyện Baseline Model -> Xuất baseline_weights (.json, .npz, .txt)
    2. Huấn luyện Best Model (tối ưu siêu tham số, early stopping, threshold sweep)
       -> Xuất best_model_weights (.json)
    
    Quy trình kết thúc sau 1 lượt chạy gọn ghẽ, KHÔNG chạy lặp liên tục.
    Tạo cấu trúc thư mục weights/ đúng y hệt hình ảnh yêu cầu:
        weights/
        ├── baseline_weights.json
        ├── baseline_weights.npz
        ├── baseline_weights.txt
        └── best_model_weights.json
    """
    import pandas as pd

    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(current_dir)

    print("=" * 80)
    print("   HỆ THỐNG HUẤN LUYỆN & XUẤT TRỌNG SỐ MÔ HÌNH HGB (ZERO SCIKIT-LEARN)")
    print("   Chế độ: Đơn lượt có kiểm soát (Tuyệt đối không chạy vòng lặp liên tục)")
    print("=" * 80)

    # 1. Tìm đường dẫn dữ liệu
    data_path = os.path.join(current_dir, "data", "SUSY.csv")
    if not os.path.exists(data_path):
        alt_path = os.path.join(current_dir, "SUSY.csv")
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy tập dữ liệu SUSY.csv tại: {data_path}")

    print(f"\n[1] Đang nạp dữ liệu từ: {data_path} ...")
    t0_data = time.time()
    df = pd.read_csv(data_path, header=None)
    y = df.iloc[:, 0].values.astype(np.int32)
    X = df.iloc[:, 1:].values.astype(np.float32)
    n_total = len(y)
    print(f"    Số lượng mẫu: {n_total:,} mẫu | Số đặc trưng: {X.shape[1]} | Thời gian đọc: {time.time()-t0_data:.2f}s")

    # 2. Phân chia Train/Test phân tầng (80/20)
    print("\n[2] Phân chia Stratified Train/Test (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train size: {len(y_train):,} mẫu | Test size: {len(y_test):,} mẫu")

    # Xác định thư mục weights mục tiêu (cả trong HGB/weights và workspace/weights)
    target_dirs = [
        os.path.join(current_dir, "weights"),
        os.path.join(workspace_dir, "weights")
    ]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)

    # --------------------------------------------------------------------------
    # 3. HUẤN LUYỆN BASELINE MODEL
    # --------------------------------------------------------------------------
    print("\n[3] Huấn luyện BASELINE MODEL (n_estimators=30, depth=5, lr=0.1, L2=1.0)...")
    t0_base = time.time()
    baseline_clf = CustomHistGradientBoostingClassifier(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        validation_fraction=0.0,
        random_state=42
    )
    baseline_clf.fit(X_train, y_train, verbose=False)
    t_base_fit = time.time() - t0_base

    # Đánh giá Baseline trên Test Set
    p_base_test = baseline_clf.predict_proba(X_test)
    y_base_pred = (p_base_test >= 0.50).astype(int)
    base_acc  = compute_accuracy(y_test, y_base_pred)
    base_prec = compute_precision(y_test, y_base_pred)
    base_rec  = compute_recall(y_test, y_base_pred)
    base_spec = compute_specificity(y_test, y_base_pred)
    base_npv  = compute_npv(y_test, y_base_pred)
    base_f1   = compute_f1_score(y_test, y_base_pred)
    base_auc  = compute_roc_auc(y_test, p_base_test)
    tp_b, tn_b, fp_b, fn_b = compute_confusion_matrix(y_test, y_base_pred)

    base_metrics = {
        "accuracy": float(base_acc),
        "precision": float(base_prec),
        "recall": float(base_rec),
        "specificity": float(base_spec),
        "npv": float(base_npv),
        "f1_score": float(base_f1),
        "roc_auc": float(base_auc),
        "tp": int(tp_b), "tn": int(tn_b), "fp": int(fp_b), "fn": int(fn_b),
        "fit_time": float(t_base_fit)
    }
    print(f"    [Baseline Test Result] Acc: {base_acc*100:.2f}% | F1: {base_f1*100:.2f}% | ROC-AUC: {base_auc:.4f} (Fit: {t_base_fit:.2f}s)")

    # Xuất các file baseline_weights
    for d in target_dirs:
        print(f"    --> Đang lưu baseline trọng số vào: {d}")
        export_weights_to_json(
            baseline_clf, os.path.join(d, "baseline_weights.json"),
            feature_names=FEATURE_NAMES, optimal_threshold=0.50,
            metrics=base_metrics, model_type="baseline"
        )
        export_weights_to_npz(
            baseline_clf, os.path.join(d, "baseline_weights.npz"),
            feature_names=FEATURE_NAMES, optimal_threshold=0.50,
            metrics=base_metrics
        )
        export_weights_to_txt(
            baseline_clf, os.path.join(d, "baseline_weights.txt"),
            feature_names=FEATURE_NAMES, optimal_threshold=0.50,
            metrics=base_metrics, model_type="baseline"
        )

    # --------------------------------------------------------------------------
    # 4. HUẤN LUYỆN BEST MODEL CHO DỰ ÁN (CHỌN CẤU HÌNH TỐI ƯU NHẤT)
    # --------------------------------------------------------------------------
    print("\n[4] Huấn luyện BEST MODEL (Đánh giá cấu hình tối ưu trên Validation set)...")
    t0_best = time.time()

    # Tách 15% train ra làm Validation để chọn cấu hình tốt nhất một cách khách quan
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split_stratified(
        X_train, y_train, test_size=0.15, random_state=42
    )

    candidate_configs = [
        {"n_estimators": 50, "learning_rate": 0.08, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 0.5},
        {"n_estimators": 60, "learning_rate": 0.08, "max_depth": 4, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"n_estimators": 70, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 0.5},
        {"n_estimators": 50, "learning_rate": 0.10, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 1.0},
        {"n_estimators": 60, "learning_rate": 0.10, "max_depth": 4, "min_samples_leaf": 25, "l2_regularization": 1.0},
    ]

    # NOTE: best_candidate_model intentionally NOT stored — the winner is
    # retrained below on the full X_train for maximum data efficiency.
    best_candidate_params = None
    best_candidate_score = -1.0
    best_thr = 0.50

    for cfg in candidate_configs:
        cand_clf = CustomHistGradientBoostingClassifier(
            **cfg,
            max_bins=255,
            validation_fraction=0.0,
            random_state=42
        )
        cand_clf.fit(X_tr_sub, y_tr_sub, verbose=False)
        p_val = cand_clf.predict_proba(X_val)
        val_auc = compute_roc_auc(y_val, p_val)

        # Quét ngưỡng tối ưu theo F1-Score trên Val
        cand_best_f1, cand_best_thr = 0.0, 0.50
        for cand_t in np.arange(0.35, 0.65, 0.05):
            f1_t = compute_f1_score(y_val, (p_val >= cand_t).astype(int))
            if f1_t > cand_best_f1:
                cand_best_f1 = f1_t
                cand_best_thr = float(cand_t)

        combined_score = val_auc + cand_best_f1
        if combined_score > best_candidate_score:
            best_candidate_score = combined_score
            best_candidate_params = cfg
            best_thr = cand_best_thr

    if best_candidate_params is None:
        raise RuntimeError("[LỖI] Không có cấu hình nào được đánh giá trong candidate_configs. Kiểm tra lại danh sách.")

    print(f"    Cấu hình tốt nhất được chọn: {best_candidate_params}")
    print(f"    Ngưỡng tối ưu khóa trên Validation: tau* = {best_thr:.2f}")

    # Huấn luyện lại trên toàn bộ X_train với cấu hình tốt nhất
    best_clf = CustomHistGradientBoostingClassifier(
        **best_candidate_params,
        max_bins=255,
        validation_fraction=0.0,
        random_state=42
    )
    best_clf.fit(X_train, y_train, verbose=False)
    t_best_fit = time.time() - t0_best

    # Đánh giá Best Model trên Test Set tại ngưỡng tối ưu
    p_best_test = best_clf.predict_proba(X_test)
    y_best_pred = (p_best_test >= best_thr).astype(int)
    best_acc  = compute_accuracy(y_test, y_best_pred)
    best_prec = compute_precision(y_test, y_best_pred)
    best_rec  = compute_recall(y_test, y_best_pred)
    best_spec = compute_specificity(y_test, y_best_pred)
    best_npv  = compute_npv(y_test, y_best_pred)
    best_f1   = compute_f1_score(y_test, y_best_pred)
    best_auc  = compute_roc_auc(y_test, p_best_test)
    tp_opt, tn_opt, fp_opt, fn_opt = compute_confusion_matrix(y_test, y_best_pred)

    best_metrics = {
        "accuracy": float(best_acc),
        "precision": float(best_prec),
        "recall": float(best_rec),
        "specificity": float(best_spec),
        "npv": float(best_npv),
        "f1_score": float(best_f1),
        "roc_auc": float(best_auc),
        "tp": int(tp_opt), "tn": int(tn_opt), "fp": int(fp_opt), "fn": int(fn_opt),
        "optimal_threshold": float(best_thr),
        "stopped_iter": int(getattr(best_clf, "stopped_iter_", len(best_clf.trees))),
        "best_n_iter": int(getattr(best_clf, "best_n_iter_", len(best_clf.trees))),
        "fit_time": float(t_best_fit)
    }
    print(f"    [Best Model Test Result] Acc: {best_acc*100:.2f}% | F1: {best_f1*100:.2f}% | ROC-AUC: {best_auc:.4f} (Fit: {t_best_fit:.2f}s)")

    # Xuất file best_model_weights.json
    for d in target_dirs:
        print(f"    --> Đang lưu best model trọng số vào: {d}")
        export_weights_to_json(
            best_clf, os.path.join(d, "best_model_weights.json"),
            feature_names=FEATURE_NAMES, optimal_threshold=best_thr,
            metrics=best_metrics, model_type="best_model"
        )

    print("\n" + "=" * 80)
    print("   HOÀN THÀNH QUY TRÌNH XUẤT TRỌNG SỐ (THÀNH CÔNG 100%)")
    print("   Thư mục trọng số đã được tạo theo đúng mẫu yêu cầu:")
    print("   weights/")
    print("   ├── baseline_weights.json")
    print("   ├── baseline_weights.npz")
    print("   ├── baseline_weights.txt")
    print("   └── best_model_weights.json")
    print("=" * 80)


if __name__ == '__main__':
    run_weights_pipeline()
