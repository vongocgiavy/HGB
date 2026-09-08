"""
Module hgb_model: Thuật toán Histogram Gradient Boosting (HGB) phân loại nhị phân,
triển khai 100% bằng Python thuần và NumPy (Zero Scikit-Learn).
Hỗ trợ đầy đủ: HistBinMapper, HistRegressionTree, CustomHistGradientBoostingClassifier,
StratifiedKFold, cross_val_score, CustomGridSearchCV và bộ metrics đánh giá.
"""

import time
import copy
import itertools
import numpy as np



# ==============================================================================
# PHẦN 1: HÀM PHÂN CHIA DỮ LIỆU & CHỈ SỐ ĐÁNH GIÁ THUẦN NUMPY (ZERO SKLEARN)
# ==============================================================================

def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Phân chia tập dữ liệu thành Train/Test phân tầng (Stratified),
    bảo toàn đúng tỷ lệ nhãn ở cả hai tập. Không dùng scikit-learn.

    Tham số:
        X            : DataFrame hoặc ndarray, ma trận đặc trưng.
        y            : ndarray, nhãn nhị phân {0, 1}.
        test_size    : float, tỷ lệ dành cho tập Test (mặc định 0.2).
        random_state : int, hạt giống ngẫu nhiên (mặc định 42).

    Trả về:
        X_train, X_test, y_train, y_test (đều là ndarray)
    """
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y)

    train_idx, test_idx = [], []
    for cls in np.unique(y_arr):
        cls_indices = np.where(y_arr == cls)[0]
        rng.shuffle(cls_indices)
        n_test = int(np.round(len(cls_indices) * test_size))
        test_idx.extend(cls_indices[:n_test])
        train_idx.extend(cls_indices[n_test:])

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    if hasattr(X, 'iloc'):
        X_train = X.iloc[train_idx].values
        X_test  = X.iloc[test_idx].values
    else:
        X_train, X_test = X[train_idx], X[test_idx]

    return X_train, X_test, y_arr[train_idx], y_arr[test_idx]


def compute_confusion_matrix(y_true, y_pred):
    """
    Ma trận nhầm lẫn nhị phân (tối ưu hóa đơn vòng bincount - Zero Sklearn).

    Trả về: (TP, TN, FP, FN)
        TP = True Positive  : thực tế 1, dự đoán 1
        TN = True Negative  : thực tế 0, dự đoán 0
        FP = False Positive : thực tế 0, dự đoán 1  (báo động giả)
        FN = False Negative : thực tế 1, dự đoán 0  (bỏ sót tín hiệu)
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    if len(y_t) != len(y_p):
        raise ValueError(f"Kích thước không khớp: y_true ({len(y_t)}) != y_pred ({len(y_p)}).")
    if not np.all(np.isin(y_t, [0, 1])):
        raise ValueError(f"y_true phải là nhãn nhị phân {{0, 1}}. Nhận được nhãn: {np.unique(y_t).tolist()}")
    if not np.all(np.isin(y_p, [0, 1])):
        raise ValueError(f"y_pred phải là nhãn nhị phân {{0, 1}}. Nhận được nhãn: {np.unique(y_p).tolist()}")

    idx = 2 * y_t.astype(int) + y_p.astype(int)
    counts = np.bincount(idx, minlength=4)
    return int(counts[3]), int(counts[0]), int(counts[1]), int(counts[2])


def compute_accuracy(y_true, y_pred):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Tỷ lệ phân loại đúng trên tổng số mẫu.
    """
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def compute_precision(y_true, y_pred):
    """
    Precision = TP / (TP + FP)
    Tỷ lệ dự đoán đúng trong số các mẫu được dự đoán là dương tính.
    """
    tp, _, fp, _ = compute_confusion_matrix(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def compute_recall(y_true, y_pred):
    """
    Recall (Sensitivity) = TP / (TP + FN)
    Tỷ lệ phát hiện đúng trong số các mẫu thực tế là dương tính.
    """
    tp, _, _, fn = compute_confusion_matrix(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def compute_f1_score(y_true, y_pred):
    """
    F1-Score = 2 * Precision * Recall / (Precision + Recall)
    Trung bình điều hòa giữa Precision và Recall.
    """
    p = compute_precision(y_true, y_pred)
    r = compute_recall(y_true, y_pred)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def compute_specificity(y_true, y_pred):
    """
    Specificity (True Negative Rate) = TN / (TN + FP)
    Tỷ lệ nhận diện chính xác các sự kiện nền (background).
    """
    _, tn, fp, _ = compute_confusion_matrix(y_true, y_pred)
    denom = tn + fp
    return tn / denom if denom > 0 else 0.0


def compute_npv(y_true, y_pred):
    """
    Negative Predictive Value = TN / (TN + FN)
    Tỷ lệ thực sự là nền trong số các mẫu dự đoán là nền.
    """
    _, tn, _, fn = compute_confusion_matrix(y_true, y_pred)
    denom = tn + fn
    return tn / denom if denom > 0 else 0.0


def compute_roc_auc(y_true, y_scores):
    """
    ROC-AUC theo công thức thống kê Mann-Whitney U / Wilcoxon rank-sum.
    AUC = P(score(pos) > score(neg)) - độ chính xác 100% so với tích phân lý thuyết.

    Tham số:
        y_true   : nhãn nhị phân {0, 1}
        y_scores : điểm số xác suất liên tục (predict_proba)

    Trả về: float trong [0, 1]
    """
    y_true   = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    if len(y_true) != len(y_scores):
        raise ValueError(f"Kích thước không khớp: y_true ({len(y_true)}) != y_scores ({len(y_scores)}).")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(f"y_true phải là nhãn nhị phân {{0, 1}}. Nhận được nhãn: {np.unique(y_true).tolist()}")
    if not np.isfinite(y_scores).all():
        raise ValueError("y_scores chứa giá trị NaN hoặc Inf.")

    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)

    # Xử lý tie (các điểm số bằng nhau): gán rank trung bình
    sorted_scores = y_scores[order]
    if np.any(sorted_scores[1:] == sorted_scores[:-1]):
        _, inv_idx, counts = np.unique(y_scores, return_inverse=True, return_counts=True)
        tie_ranks = np.cumsum(np.r_[0, counts[:-1]]) + (counts + 1) / 2.0
        ranks = tie_ranks[inv_idx]

    sum_pos_ranks = np.sum(ranks[pos_mask])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_roc_curve(y_true, y_scores, drop_intermediate: bool = True):
    """
    Tính False Positive Rate (FPR) và True Positive Rate (TPR) qua các ngưỡng phân loại (Zero Sklearn).

    Tham số:
        y_true            : Nhãn thực tế nhị phân {0, 1}
        y_scores          : Điểm số xác suất liên tục
        drop_intermediate : bool, loại bỏ các điểm cận tối ưu trên đoạn đồng tuyến (chuẩn Scikit-Learn)
    Trả về:
        fpr, tpr, thresholds (np.ndarray)
    """
    y_t = np.asarray(y_true).ravel()
    scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if len(y_t) != len(scores):
        raise ValueError(f"Kích thước không khớp: y_true ({len(y_t)}) != y_scores ({len(scores)}).")
    if not np.all(np.isin(y_t, [0, 1])):
        raise ValueError(f"y_true phải là nhãn nhị phân {{0, 1}}.")
    if not np.isfinite(scores).all():
        raise ValueError("y_scores chứa giá trị NaN hoặc Inf.")

    y_t = y_t.astype(int)
    desc_order = np.argsort(-scores)
    y_sorted = y_t[desc_order]
    scores_sorted = scores[desc_order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_t.size - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))

    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        threshold_idxs = threshold_idxs[optimal_idxs]

    tpr = tps / max(n_pos, 1)
    fpr = fps / max(n_neg, 1)

    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thresholds = np.r_[np.inf, scores_sorted[threshold_idxs]]
    return fpr, tpr, thresholds


def compute_precision_recall_curve(y_true, y_scores):
    """
    Tính Precision và Recall qua các ngưỡng phân loại tăng dần (chuẩn Scikit-Learn API).

    Tham số:
        y_true   : Nhãn thực tế nhị phân {0, 1}
        y_scores : Điểm số xác suất liên tục
    Trả về:
        precision, recall, thresholds (np.ndarray)
        - precision : mảng độ dài K + 1 (phần tử cuối là 1.0)
        - recall    : mảng độ dài K + 1 (phần tử cuối là 0.0)
        - thresholds: mảng độ dài K (thứ tự tăng dần)
    """
    y_t = np.asarray(y_true).ravel()
    scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if len(y_t) != len(scores):
        raise ValueError(f"Kích thước không khớp: y_true ({len(y_t)}) != y_scores ({len(scores)}).")
    if not np.all(np.isin(y_t, [0, 1])):
        raise ValueError(f"y_true phải là nhãn nhị phân {{0, 1}}.")
    if not np.isfinite(scores).all():
        raise ValueError("y_scores chứa giá trị NaN hoặc Inf.")

    y_t = y_t.astype(int)
    desc_order = np.argsort(-scores)
    y_sorted = y_t[desc_order]
    scores_sorted = scores[desc_order]

    distinct_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_t.size - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    n_pos = int(np.sum(y_t == 1))
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(n_pos, 1)

    # Đảo chiều để thresholds tăng dần theo chuẩn Scikit-Learn
    # Đảm bảo: len(precision) == len(recall) == len(thresholds) + 1
    p = np.r_[precision[::-1], 1.0]
    r = np.r_[recall[::-1], 0.0]
    th = scores_sorted[threshold_idxs][::-1]
    return p, r, th


# Aliases tương thích với Scikit-Learn
roc_curve = compute_roc_curve
precision_recall_curve = compute_precision_recall_curve


# ==============================================================================
# PHẦN 2: HISTOGRAM BIN MAPPER
# ==============================================================================

class HistBinMapper:
    """
    Rời rạc hóa ma trận đặc trưng liên tục thành thùng số nguyên uint8
    bằng phương pháp phân vị đều (Quantile Binning).

    Phương thức:
        fit(X)       : Tính ngưỡng phân vị trên tập huấn luyện.
        transform(X) : Ánh xạ X thành ma trận uint8 chỉ số bin.

    Lưu ý: Chỉ gọi fit() trên X_train. Tuyệt đối không fit trên X_test
            để tránh data leakage phân phối.
    """

    def __init__(self, max_bins: int = 255):
        """Khởi tạo HistBinMapper với số thùng tối đa (mặc định 255, tối đa 256 do kiểu uint8)."""
        max_bins_int = int(max_bins)
        if not (2 <= max_bins_int <= 256):
            raise ValueError(
                f"max_bins phai nam trong khoang [2, 256] do bieu dien uint8 [0..255]. "
                f"Nhan duoc: {max_bins}"
            )
        self.max_bins = max_bins_int
        self.bin_thresholds_ = []   # Danh sách ngưỡng cho từng đặc trưng
        self.n_features_in_ = None

    def fit(self, X) -> "HistBinMapper":
        """
        Tính ngưỡng phân vị đều (quantile) cho mỗi đặc trưng trên X_train.
        Chỉ gọi fit() trên tập huấn luyện để tránh data leakage.
        Trả về: self (để chain .fit().transform())
        """
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("Dữ liệu X chứa giá trị không hợp lệ (NaN hoặc +/-Inf). Vui lòng xử lý (impute/clean) trước khi fit.")
        
        self.n_features_in_ = X.shape[1]
        self.bin_thresholds_ = []
        percentiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]

        for j in range(self.n_features_in_):
            thresholds = np.nanpercentile(X[:, j], percentiles)
            thresholds = np.unique(thresholds)
            self.bin_thresholds_.append(thresholds)
        return self

    def transform(self, X) -> np.ndarray:
        """Trả về ma trận uint8 có cùng shape với X."""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Số đặc trưng không khớp: fit với {self.n_features_in_} cột, "
                f"nhưng transform nhận {X.shape[1]} cột."
            )
        if not np.isfinite(X).all():
            raise ValueError("Dữ liệu X truyền vào transform chứa giá trị NaN hoặc +/-Inf.")

        n_samples, n_features = X.shape
        X_binned = np.empty((n_samples, n_features), dtype=np.uint8)

        for j in range(n_features):
            # np.searchsorted: O(N log K) ánh xạ nhanh vào chỉ số bin
            X_binned[:, j] = np.searchsorted(
                self.bin_thresholds_[j], X[:, j], side='right'
            ).astype(np.uint8)
        return X_binned


# ==============================================================================
# PHẦN 3: CÂY QUYẾT ĐỊNH HISTOGRAM
# ==============================================================================

class HistTreeNode:
    """
    Một nút trong cây quyết định Histogram.

    Thuộc tính:
        is_leaf        : bool   - Nút lá hay nút rẽ nhánh.
        value          : float  - Trọng số lá w* (chỉ có ý nghĩa nếu is_leaf=True).
        feature_idx    : int    - Chỉ số đặc trưng dùng để chia (nút rẽ nhánh).
        bin_threshold  : int    - Ngưỡng bin uint8 để chia nhánh trái/phải.
        gain           : float  - Độ lợi phân tách đạt được.
        left, right    : HistTreeNode - Nút con trái và phải.
    """
    __slots__ = ('is_leaf', 'value', 'feature_idx', 'bin_threshold', 'gain', 'left', 'right')

    def __init__(self, is_leaf=False, value=0.0, feature_idx=None,
                 bin_threshold=None, gain=0.0):
        """
        Khởi tạo nút cây Histogram.
            is_leaf       : True nếu là nút lá, False nếu là nút rẽ nhánh.
            value         : Trọng số lá w* (chỉ dùng khi is_leaf=True).
            feature_idx   : Chỉ số đặc trưng dùng để chia (nút rẽ nhánh).
            bin_threshold : Ngưỡng bin uint8 — đi trái nếu X[feat] <= threshold.
            gain          : Độ lợi phân tách Gain đạt được tại nút này.
        """
        self.is_leaf       = is_leaf
        self.value         = value
        self.feature_idx   = feature_idx
        self.bin_threshold = bin_threshold
        self.gain          = gain
        self.left          = None
        self.right         = None


class HistRegressionTree:
    """
    Cây hồi quy Histogram tối ưu hóa tổng hàm mất mát dựa trên
    gradient bậc 1 (g) và hessian bậc 2 (h).

    Thuật toán tìm điểm chia nhanh O(D * K) thông qua:
      - np.bincount: xây histogram G_k, H_k, Count_k
      - np.cumsum  : tính G_L, H_L, Count_L tích lũy bên trái

    Tham số:
        max_depth        : int   - Độ sâu tối đa của cây.
        min_samples_leaf : int   - Số mẫu tối thiểu ở mỗi nút lá.
        l2_regularization: float - Hệ số phạt L2 (lambda) trong công thức Gain và w*.
        min_gain_to_split: float - Ngưỡng Gain tối thiểu (gamma) để tạo nhánh.
        max_bins         : int   - Kích thước histogram (phải khớp với HistBinMapper).
    """

    def __init__(self, max_depth=6, min_samples_leaf=20,
                 l2_regularization=1.0, min_gain_to_split=1e-7, max_bins=255):
        """
        Khởi tạo cây hồi quy Histogram.
            max_depth        : Độ sâu tối đa (mặc định 6).
            min_samples_leaf : Mẫu tối thiểu ở nút lá (mặc định 20).
            l2_regularization: Hệ số phạt L2 lambda (mặc định 1.0).
            min_gain_to_split: Ngưỡng Gain tối thiểu gamma (mặc định 1e-7).
            max_bins         : Kích thước histogram — phải khớp HistBinMapper (mặc định 255).
        """
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_reg           = l2_regularization
        self.min_gain         = min_gain_to_split
        self.max_bins         = max_bins
        self.root             = None

    # ---- Huấn luyện ----

    def fit(self, X_binned: np.ndarray, g: np.ndarray, h: np.ndarray) -> "HistRegressionTree":
        """
        Xây dựng cây dựa trên:
            X_binned : (N, D) uint8 - Đặc trưng đã rời rạc hóa
            g        : (N,)  float  - Gradient bậc 1  g_i = p_i - y_i
            h        : (N,)  float  - Hessian bậc 2   h_i = p_i * (1 - p_i)
        """
        indices = np.arange(len(g))
        self.root = self._build(X_binned, g, h, indices, depth=0)
        return self

    def _leaf_weight(self, g, h, idx):
        """
        Trọng số lá Newton-Raphson có phạt L2:
        w* = -sum(g_i) / (sum(h_i) + lambda)
        """
        return -np.sum(g[idx]) / (np.sum(h[idx]) + self.l2_reg)

    def _best_split(self, X_binned, g, h, idx):
        """
        Duyệt toàn bộ D đặc trưng và K bins để tìm điểm cắt
        tối đa hóa Gain phân tách (được tối ưu hóa bằng vector hóa đơn vòng).

        Công thức Gain:
            Gain = 0.5 * [ G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G_tot^2/(H_tot+λ) ]
        """
        n = len(idx)
        if n < 2 * self.min_samples_leaf:
            return None

        g_node, h_node = g[idx], h[idx]
        G_tot, H_tot   = float(g_node.sum()), float(h_node.sum())
        score_tot      = G_tot ** 2 / (H_tot + self.l2_reg)
        best_split_score = score_tot + 2.0 * self.min_gain
        best_feat, best_bin = None, None

        for j in range(X_binned.shape[1]):
            col = X_binned[idx, j]
            G_bin = np.bincount(col, weights=g_node, minlength=self.max_bins)
            H_bin = np.bincount(col, weights=h_node, minlength=self.max_bins)
            C_bin = np.bincount(col,                 minlength=self.max_bins)

            G_L = np.cumsum(G_bin)[:-1]
            H_L = np.cumsum(H_bin)[:-1]
            C_L = np.cumsum(C_bin)[:-1]
            G_R = G_tot - G_L
            H_R = H_tot - H_L
            C_R = n - C_L

            valid = (C_L >= self.min_samples_leaf) & (C_R >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            valid_bins = np.flatnonzero(valid)
            score_split = (
                G_L[valid_bins] ** 2 / (H_L[valid_bins] + self.l2_reg) +
                G_R[valid_bins] ** 2 / (H_R[valid_bins] + self.l2_reg)
            )
            k = score_split.argmax()
            if score_split[k] > best_split_score:
                best_split_score = score_split[k]
                best_feat = j
                best_bin  = int(valid_bins[k])

        if best_feat is not None:
            best_gain = float(0.5 * (best_split_score - score_tot))
            return best_feat, best_bin, best_gain
        return None

    def _build(self, X_binned, g, h, idx, depth):
        """Đệ quy xây cây nhị phân."""
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return HistTreeNode(is_leaf=True, value=self._leaf_weight(g, h, idx))

        split = self._best_split(X_binned, g, h, idx)
        if split is None:
            return HistTreeNode(is_leaf=True, value=self._leaf_weight(g, h, idx))

        feat, bin_thr, gain = split
        mask = X_binned[idx, feat] <= bin_thr
        node = HistTreeNode(is_leaf=False, feature_idx=feat,
                            bin_threshold=bin_thr, gain=gain)
        node.left  = self._build(X_binned, g, h, idx[mask],  depth + 1)
        node.right = self._build(X_binned, g, h, idx[~mask], depth + 1)
        return node

    # ---- Dự đoán ----

    def predict(self, X_binned: np.ndarray) -> np.ndarray:
        """Trả về mảng float32 trọng số lá cho mỗi mẫu."""
        preds = np.zeros(X_binned.shape[0], dtype=np.float32)
        self._traverse(self.root, X_binned, np.arange(len(preds)), preds)
        return preds

    def _traverse(self, node, X_binned, idx, preds):
        """
        Duyệt đệ quy cây từ node cho tập chỉ mục idx,
        ghi trọng số lá vào mảng preds tại đúng vị trí.
        """
        if len(idx) == 0:
            return
        if node.is_leaf:
            preds[idx] = node.value
            return
        mask = X_binned[idx, node.feature_idx] <= node.bin_threshold
        self._traverse(node.left,  X_binned, idx[mask],  preds)
        self._traverse(node.right, X_binned, idx[~mask], preds)

    def compute_feature_importances(self, n_features: int) -> np.ndarray:
        """Tính tổng Gain tích lũy cho từng đặc trưng trong cây."""
        importances = np.zeros(n_features, dtype=np.float64)

        def _traverse_gain(node):
            if node is None or node.is_leaf:
                return
            if node.feature_idx is not None and node.gain > 0:
                importances[node.feature_idx] += float(node.gain)
            _traverse_gain(node.left)
            _traverse_gain(node.right)

        _traverse_gain(self.root)
        return importances


# ==============================================================================
# PHẦN 4: BỘ PHÂN LOẠI HGB CHÍNH (CLASS CHỦ ĐẠO)
# ==============================================================================

class CustomHistGradientBoostingClassifier:
    """
    Bộ phân loại Histogram Gradient Boosting (HGB) cho phân loại nhị phân (Zero Sklearn).
    
    Hỗ trợ: Rời rạc hóa phân vị (uint8), tối ưu hóa Newton-Raphson bậc 2 với Gradient/Hessian,
    tìm điểm cắt O(D*K) tích lũy, điều chuẩn L2 trên trọng số lá, và Early Stopping nội bộ.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_bins=255,
        min_gain_to_split=0.0,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4,
        random_state=42
    ):
        """Khởi tạo siêu tham số cho bộ phân loại HGB."""
        self.n_estimators        = int(n_estimators)
        self.learning_rate       = float(learning_rate)
        self.max_depth           = int(max_depth)
        self.min_samples_leaf    = int(min_samples_leaf)
        self.l2_reg              = float(l2_regularization)
        self.l2_regularization   = float(l2_regularization)
        self.max_bins            = int(max_bins)
        self.min_gain            = float(min_gain_to_split)
        self.min_gain_to_split   = float(min_gain_to_split)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change    = int(n_iter_no_change)
        self.tol                 = float(tol)
        self.random_state        = random_state

        self._validate_params()

        # --- Trạng thái nội bộ ---
        self.bin_mapper          = HistBinMapper(max_bins=self.max_bins)
        self.trees               = []
        self.base_score_         = 0.0
        self.n_features_in_      = None
        self.classes_            = np.array([0, 1], dtype=np.int32)
        self.n_iter_             = 0
        self.best_n_iter_        = 0
        self.best_val_loss_      = np.inf
        self.feature_importances_ = None
        self.train_loss_history_       = []
        self.val_loss_history_         = []
        self.full_train_loss_history_  = []   # lịch sử đầy đủ đến khi early-stop (trước khi trim)
        self.full_val_loss_history_    = []   # lịch sử đầy đủ đến khi early-stop (trước khi trim)

    def _validate_params(self):
        """Kiem tra tinh hop le cua toan bo sieu tham so mo hinh."""
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators phai la so nguyen duong > 0 (nhan duoc: {self.n_estimators}).")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate phai la so thuc duong > 0 (nhan duoc: {self.learning_rate}).")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth phai la so nguyen duong > 0 (nhan duoc: {self.max_depth}).")
        if self.min_samples_leaf <= 0:
            raise ValueError(f"min_samples_leaf phai la so nguyen duong > 0 (nhan duoc: {self.min_samples_leaf}).")
        if self.l2_reg < 0:
            raise ValueError(f"l2_regularization phai >= 0 (nhan duoc: {self.l2_reg}).")
        if not (2 <= self.max_bins <= 256):
            raise ValueError(f"max_bins phai nam trong khoang [2, 256] do kieu uint8 (nhan duoc: {self.max_bins}).")
        if self.min_gain < 0:
            raise ValueError(f"min_gain_to_split phai >= 0 (nhan duoc: {self.min_gain}).")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError(f"validation_fraction phai nam trong khoang (0.0, 1.0) (nhan duoc: {self.validation_fraction}).")
        if self.n_iter_no_change <= 0:
            raise ValueError(f"n_iter_no_change (patience) phai la so nguyen duong > 0 (nhan duoc: {self.n_iter_no_change}).")
        if self.tol < 0:
            raise ValueError(f"tol phai >= 0 (nhan duoc: {self.tol}).")

    def get_params(self, deep=True):
        """Lấy danh sách tham số (tương thích Scikit-Learn GridSearchCV & CustomGridSearchCV)."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'l2_regularization': getattr(self, 'l2_regularization', self.l2_reg),
            'max_bins': self.max_bins,
            'min_gain_to_split': getattr(self, 'min_gain_to_split', self.min_gain),
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'tol': self.tol,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        """Cập nhật siêu tham số cho mô hình."""
        for key, value in params.items():
            if key in ('l2_regularization', 'l2_reg'):
                self.l2_reg = float(value)
                self.l2_regularization = float(value)
            elif key in ('min_gain_to_split', 'min_gain'):
                self.min_gain = float(value)
                self.min_gain_to_split = float(value)
            elif key in ('n_estimators', 'max_depth', 'min_samples_leaf', 'n_iter_no_change'):
                setattr(self, key, int(value))
            elif key in ('learning_rate', 'validation_fraction', 'tol'):
                setattr(self, key, float(value))
            elif key == 'max_bins':
                self.max_bins = int(value)
                self.bin_mapper = HistBinMapper(max_bins=self.max_bins)
            else:
                setattr(self, key, value)
        self._validate_params()
        return self

    # ---------- Hàm tĩnh hỗ trợ ----------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid p = 1 / (1 + exp(-x)), clip [-15, 15] để tránh overflow."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    @staticmethod
    def _log_loss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy Loss:
        L = -mean( y*log(p) + (1-y)*log(1-p) )
        """
        p = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    # ---------- Tách Validation Set nội bộ ----------

    def _split_validation(self, X, y):
        """
        Tách Validation Set phân tầng từ dữ liệu train RAW (float32 — chưa binned).
        Hàm trả về chỉ mục để fit() có thể:
          1. fit bin_mapper CHỈ trên tập train-sub (tránh leakage phân phối).
          2. transform riêng X_raw_tr và X_raw_val.
        KHÔNG dùng tập Test ngoài → đảm bảo không có data leakage.
        """
        rng = np.random.RandomState(self.random_state)
        val_idx, tr_idx = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            n_val = max(1, int(np.round(len(idx) * self.validation_fraction)))
            val_idx.extend(idx[:n_val])
            tr_idx.extend(idx[n_val:])
        tr_idx  = np.array(tr_idx)
        val_idx = np.array(val_idx)
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    # ---------- Huấn luyện ----------

    def fit(self, X, y, verbose=False) -> "CustomHistGradientBoostingClassifier":
        """
        Huấn luyện mô hình HGB với Early Stopping.

        Quy trình:
            1. Kiểm tra tính toàn vẹn dữ liệu (NaN, chiều đặc trưng, nhãn nhị phân).
            2. Tách 10% thành Validation Set nội bộ RAW (phân tầng) — trước khi binning.
            3. bin_mapper.fit(X_train_sub) → rời rạc hóa CHỈ trên train-sub (tránh leakage).
               transform X_train_sub và X_val riêng biệt.
            4. Khởi tạo F_0 = log-odds trên tập train-sub.
            5. Lặp m = 1..n_estimators:
               a. Tính g = p - y,  h = p*(1-p)  trên tập train-sub.
               b. Fit HistRegressionTree T_m trên (g, h).
               c. Cập nhật F_train và F_val bằng shrinkage.
               d. Tính Train Loss và Val Loss.
               e. Nếu Val Loss không giảm thêm tol sau n_iter_no_change vòng → dừng.
            6. Trim self.trees, train_loss_history_, val_loss_history_ về best_n_iter_.
               Lịch sử đầy đủ được lưu trong full_train/val_loss_history_.
        """
        # Kiểm tra tính hợp lệ của dữ liệu đầu vào
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self._validate_params()

        if not np.isfinite(X).all():
            raise ValueError("Du lieu dau vao X chua gia tri khong hop le (NaN hoac +/-Inf). Vui long xu ly truoc khi fit().")
        if not np.isfinite(y).all():
            raise ValueError("Nhan y chua gia tri khong hop le (NaN hoac +/-Inf). Vui long xu ly truoc khi fit().")

        unique_y = np.unique(y)
        if not np.all(np.isin(unique_y, [0, 1])):
            raise ValueError(
                f"CustomHistGradientBoostingClassifier hien chi ho tro phan loai nhi phan nhan {{0, 1}}. "
                f"Nhan nhan duoc: {unique_y.tolist()}"
            )

        self.n_features_in_ = X.shape[1]

        # Reset trạng thái
        self.trees               = []
        self.train_loss_history_ = []
        self.val_loss_history_   = []
        self.best_val_loss_      = np.inf
        self.best_n_iter_        = 0
        no_improve               = 0

        # Bước 1: Tách Validation nội bộ TRƯỚC khi rời rạc hóa
        #   → bin_mapper sẽ chỉ được fit trên X_raw_tr (train-sub),
        #     không "nhìn thấy" phân phối của X_raw_val.
        #   → Đảm bảo validation nội bộ hoàn toàn độc lập về preprocessing.
        X_raw_tr, y_tr, X_raw_val, y_val = self._split_validation(X, y)
        if verbose:
            print(f"[*] Validation noi bo: {len(y_tr):,} train-sub | {len(y_val):,} val "
                  f"(validation_fraction={self.validation_fraction})")

        # Bước 2: Rời rạc hóa — fit CHỈ trên X_raw_tr, transform cả hai tập riêng
        if verbose:
            print(f"[*] Roi rac hoa {self.max_bins} bins (Quantile Binning - fit tren {len(X_raw_tr):,} train-sub samples only)...")
        self.bin_mapper = HistBinMapper(max_bins=self.max_bins)
        self.bin_mapper.fit(X_raw_tr)
        X_tr  = self.bin_mapper.transform(X_raw_tr)
        X_val = self.bin_mapper.transform(X_raw_val)

        # Bước 3: Khởi tạo F_0 = log-odds
        y_mean = float(np.clip(np.mean(y_tr), 1e-7, 1.0 - 1e-7))
        self.base_score_ = np.log(y_mean / (1.0 - y_mean))
        F_tr  = np.full(len(y_tr),  self.base_score_, dtype=np.float32)
        F_val = np.full(len(y_val), self.base_score_, dtype=np.float32)

        if verbose:
            print(f"[*] F_0 (log-odds) = {self.base_score_:.4f}")
            print(f"[*] Boosting toi da {self.n_estimators} vong | "
                  f"Early stopping sau {self.n_iter_no_change} vong (tol={self.tol})")
            print(f"  {'Vong':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
                  f"{'Best Val':>10} | {'Thoi gian':>10}")
            print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

        t0 = time.time()
        for m in range(1, self.n_estimators + 1):
            t_iter = time.time()

            # Gradient & Hessian bậc 1 & 2
            p_tr = self._sigmoid(F_tr)
            g = p_tr - y_tr
            h = np.maximum(p_tr * (1.0 - p_tr), 1e-16)

            # Fit cây trên tập train-sub
            tree = HistRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_reg,
                min_gain_to_split=self.min_gain,
                max_bins=self.max_bins,
            )
            tree.fit(X_tr, g, h)
            self.trees.append(tree)

            # Shrinkage update
            F_tr  += self.learning_rate * tree.predict(X_tr)
            F_val += self.learning_rate * tree.predict(X_val)

            # Tính loss
            tl = self._log_loss(y_tr,  self._sigmoid(F_tr))
            vl = self._log_loss(y_val, self._sigmoid(F_val))
            self.train_loss_history_.append(tl)
            self.val_loss_history_.append(vl)

            improved = vl < (self.best_val_loss_ - self.tol)
            if improved:
                self.best_val_loss_ = vl
                self.best_n_iter_   = m
                no_improve          = 0
                flag = "(*)"
            else:
                no_improve += 1
                flag = ""

            dt = time.time() - t_iter
            if verbose and (m == 1 or m % 10 == 0 or improved or
                            no_improve == self.n_iter_no_change or m == self.n_estimators):
                print(f"  [{m:04d}] | {tl:12.5f} | {vl:12.5f} | "
                      f"{self.best_val_loss_:>10.5f} | {dt:8.2f}s  {flag}")

            if no_improve >= self.n_iter_no_change:
                if verbose:
                    print(f"\n[!] EARLY STOPPING vong {m}: "
                          f"Val Loss khong cai thien sau {self.n_iter_no_change} vong.")
                    print(f"    Val Loss tot nhat: {self.best_val_loss_:.5f} "
                          f"tai vong {self.best_n_iter_}")
                break

        self.stopped_iter_ = len(self.trees)

        # Lưu toàn bộ lịch sử loss trước khi trim (hữu ích để vẽ biểu đồ quá trình dừng)
        self.full_train_loss_history_ = list(self.train_loss_history_)
        self.full_val_loss_history_   = list(self.val_loss_history_)

        if self.best_n_iter_ is not None and self.best_n_iter_ > 0:
            self.trees               = self.trees[:self.best_n_iter_]
            # Trim loss history để khớp với số cây thực tế — tránh lệch khi vẽ biểu đồ
            self.train_loss_history_ = self.train_loss_history_[:self.best_n_iter_]
            self.val_loss_history_   = self.val_loss_history_[:self.best_n_iter_]
        self.n_iter_ = len(self.trees)
        elapsed = time.time() - t0

        # Bước 4: Tính Feature Importances (Gain-based) tích lũy trên toàn bộ cây
        raw_importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree in self.trees:
            raw_importances += tree.compute_feature_importances(self.n_features_in_)
        tot_gain = np.sum(raw_importances)
        if tot_gain > 0:
            self.feature_importances_ = (raw_importances / tot_gain).astype(np.float64)
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)

        if verbose:
            print(f"[*] Xong {self.n_iter_} cay | "
                  f"Tong: {elapsed:.2f}s ({elapsed/self.n_iter_:.3f}s/cay)\n")
        return self

    # ---------- Dự đoán ----------

    def predict_raw(self, X) -> np.ndarray:
        """Trả về điểm số logit thô F(x) cho từng mẫu."""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        if not hasattr(self, 'n_features_in_') or self.n_features_in_ is None or len(self.trees) == 0:
            raise RuntimeError("Mô hình chưa được huấn luyện. Vui lòng gọi fit(X, y) trước khi predict.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Số đặc trưng không khớp: fit với {self.n_features_in_} đặc trưng, "
                f"nhưng predict nhận {X.shape[1]} đặc trưng."
            )
        if not np.isfinite(X).all():
            raise ValueError("Dữ liệu X truyền vào predict chứa giá trị không hợp lệ (NaN hoặc +/-Inf).")

        X_b = self.bin_mapper.transform(X)
        F   = np.full(X_b.shape[0], self.base_score_, dtype=np.float32)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X_b)
        return F

    def predict_proba(self, X) -> np.ndarray:
        """Trả về xác suất p = sigmoid(F(x)) cho từng mẫu."""
        return self._sigmoid(self.predict_raw(X))

    def predict_proba_2d(self, X) -> np.ndarray:
        """Trả về mảng xác suất 2 chiều [P(y=0), P(y=1)] tương thích chuẩn Scikit-Learn."""
        p1 = self.predict_proba(X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold: float = 0.40) -> np.ndarray:
        """Trả về nhãn nhị phân {0, 1} theo ngưỡng phân loại threshold (mặc định: 0.40)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y) -> float:
        """Độ chính xác (Accuracy) mặc định trên tập kiểm tra (X, y)."""
        preds = self.predict(X, threshold=0.40)
        return compute_accuracy(y, preds)


# ==============================================================================
# PHẦN 5: KIỂM TRA CHÉO (CROSS-VALIDATION) & GRID SEARCH CV (ZERO SKLEARN)
# ==============================================================================

class StratifiedKFold:
    """
    K-Fold phân tầng (Stratified K-Fold) tự viết 100% bằng NumPy - Zero Scikit-Learn.
    Chia dữ liệu thành k folds sao cho tỷ lệ các lớp nhãn {0, 1} được bảo toàn đồng đều trong mỗi fold.

    Tham số:
        n_splits     (int)  : Số lượng folds (mặc định 5).
        shuffle      (bool) : Có xáo trộn ngẫu nhiên trước khi chia không (mặc định True).
        random_state (int)  : Hạt giống ngẫu nhiên (mặc định 42).
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        """
        Khởi tạo StratifiedKFold.
            n_splits     : Số folds (>= 2, mặc định 5).
            shuffle      : Xáo trộn ngẫu nhiên trước khi chia (mặc định True).
            random_state : Hạt giống ngẫu nhiên (mặc định 42).
        """
        if n_splits < 2:
            raise ValueError(f"n_splits phải >= 2, nhận {n_splits}.")
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(self, X, y):
        """
        Sinh ra các cặp chỉ mục (train_indices, val_indices) cho từng fold.
        """
        y_arr = np.asarray(y).ravel()
        n_samples = len(y_arr)
        rng = np.random.RandomState(self.random_state)

        classes = np.unique(y_arr)
        fold_val_indices = [[] for _ in range(self.n_splits)]

        for cls in classes:
            cls_idx = np.where(y_arr == cls)[0].copy()
            if self.shuffle:
                rng.shuffle(cls_idx)
            splits = np.array_split(cls_idx, self.n_splits)
            for fold_i, split_arr in enumerate(splits):
                fold_val_indices[fold_i].extend(split_arr)

        for fold_i in range(self.n_splits):
            val_idx = np.array(fold_val_indices[fold_i], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(val_idx)

            mask = np.ones(n_samples, dtype=bool)
            mask[val_idx] = False
            train_idx = np.where(mask)[0]
            if self.shuffle:
                rng.shuffle(train_idx)

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None):
        """Trả về số lượng folds."""
        return self.n_splits


def cross_val_score(
    estimator,
    X,
    y,
    cv=5,
    scoring='roc_auc',
    threshold=0.40,
    verbose=0
):
    """
    Đánh giá mô hình qua kiểm tra chéo K-Fold phân tầng (Stratified K-Fold CV).
    Thuần NumPy - không sử dụng Scikit-Learn.

    Tham số:
        estimator : Mô hình có phương thức fit, predict, predict_proba
        X         : Ma trận đặc trưng
        y         : Nhãn nhị phân {0, 1}
        cv        : Số folds int (mặc định 5) hoặc đối tượng StratifiedKFold
        scoring   : 'roc_auc' (mặc định), 'f1', 'accuracy', 'precision', 'recall' hoặc callable
        threshold : Ngưỡng phân loại dùng cho f1, accuracy, precision, recall (mặc định 0.40)
        verbose   : Mức độ in tiến trình (0: im lặng, 1: in kết quả từng fold)

    Trả về:
        scores (np.ndarray): Mảng điểm số của từng fold
    """
    if isinstance(cv, int):
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=getattr(estimator, 'random_state', 42)
        )
    else:
        cv_splitter = cv

    if hasattr(X, 'iloc'):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    scores = []
    n_splits = cv_splitter.get_n_splits()
    for fold_i, (train_idx, val_idx) in enumerate(cv_splitter.split(X_arr, y_arr)):
        t0 = time.time()
        model = copy.deepcopy(estimator)
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]

        model.fit(X_tr, y_tr, verbose=False)

        if callable(scoring):
            score = float(scoring(model, X_val, y_val))
        elif scoring == 'roc_auc':
            probs = model.predict_proba(X_val)
            if probs.ndim == 2:
                probs = probs[:, 1]
            score = float(compute_roc_auc(y_val, probs))
        elif scoring == 'f1':
            preds = model.predict(X_val, threshold=threshold)
            score = float(compute_f1_score(y_val, preds))
        elif scoring == 'accuracy':
            preds = model.predict(X_val, threshold=threshold)
            score = float(compute_accuracy(y_val, preds))
        elif scoring == 'precision':
            preds = model.predict(X_val, threshold=threshold)
            score = float(compute_precision(y_val, preds))
        elif scoring == 'recall':
            preds = model.predict(X_val, threshold=threshold)
            score = float(compute_recall(y_val, preds))
        else:
            raise ValueError(f"Chỉ số scoring '{scoring}' không được hỗ trợ. Chọn: 'roc_auc', 'f1', 'accuracy', 'precision', 'recall'.")

        scores.append(score)
        if verbose > 0:
            print(f"  [Fold {fold_i + 1}/{n_splits}] {scoring}: {score:.4f} ({time.time() - t0:.2f}s)")

    return np.array(scores, dtype=np.float64)


class CustomGridSearchCV:
    """
    Tìm kiếm lưới siêu tham số (Grid Search) kết hợp Stratified K-Fold CV (Zero Sklearn).
    Duyệt tích Descartes các tham số, tính điểm cv_results_ và refit best_estimator_.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        scoring='roc_auc',
        cv=5,
        refit=True,
        threshold=0.40,
        verbose=1
    ):
        """Khởi tạo cấu hình tìm kiếm lưới K-Fold."""
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.threshold = threshold
        self.verbose = verbose

        # Kết quả sau khi fit()
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        self.best_index_ = -1
        self.cv_results_ = {}
        self.n_splits_ = 0

    def _generate_param_candidates(self):
        """Sinh danh sách tất cả các tổ hợp siêu tham số từ param_grid."""
        keys = list(self.param_grid.keys())
        values = [
            self.param_grid[k] if isinstance(self.param_grid[k], (list, tuple, np.ndarray))
            else [self.param_grid[k]]
            for k in keys
        ]
        candidates = []
        for combo in itertools.product(*values):
            candidates.append(dict(zip(keys, combo)))
        return candidates

    def fit(self, X, y):
        """
        Thực hiện tìm kiếm lưới qua K-Fold Cross Validation trên tập (X, y).
        """
        if hasattr(X, 'iloc'):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if isinstance(self.cv, int):
            cv_splitter = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=getattr(self.estimator, 'random_state', 42)
            )
        else:
            cv_splitter = self.cv

        self.n_splits_ = cv_splitter.get_n_splits()
        candidates = self._generate_param_candidates()
        n_candidates = len(candidates)

        if self.verbose >= 1:
            print("=" * 78)
            print("[*] START CUSTOM GRID SEARCH CV (ZERO SKLEARN)")
            print(f"    - Tong so to hop tham so : {n_candidates}")
            print(f"    - So Folds kiem tra cheo : {self.n_splits_} (Stratified K-Fold)")
            print(f"    - Tong so luot train     : {n_candidates * self.n_splits_} fits")
            print(f"    - Tieu chi toi uu        : {self.scoring.upper()} (threshold={self.threshold})")
            print("=" * 78)

        # Chuẩn bị cấu trúc cv_results_
        cv_results = {
            'params': candidates,
            'mean_test_score': np.zeros(n_candidates, dtype=np.float64),
            'std_test_score': np.zeros(n_candidates, dtype=np.float64),
            'mean_fit_time': np.zeros(n_candidates, dtype=np.float64),
        }
        for f in range(self.n_splits_):
            cv_results[f'split{f}_test_score'] = np.zeros(n_candidates, dtype=np.float64)

        start_total = time.time()
        for cand_i, params in enumerate(candidates):
            t_cand_start = time.time()
            fold_scores = []
            fold_fit_times = []

            if self.verbose >= 1:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"[{cand_i + 1:02d}/{n_candidates:02d}] Thu nghiem: {{{param_str}}}")

            for fold_i, (train_idx, val_idx) in enumerate(cv_splitter.split(X_arr, y_arr)):
                t0 = time.time()
                model = copy.deepcopy(self.estimator)
                model.set_params(**params)

                X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
                X_val, y_val = X_arr[val_idx], y_arr[val_idx]

                # Fit mô hình con với verbose=False để tránh in nhiễu
                model.fit(X_tr, y_tr, verbose=False)
                fit_duration = time.time() - t0
                fold_fit_times.append(fit_duration)

                # Đánh giá điểm theo metric lựa chọn
                if callable(self.scoring):
                    score = float(self.scoring(model, X_val, y_val))
                elif self.scoring == 'roc_auc':
                    probs = model.predict_proba(X_val)
                    if probs.ndim == 2:
                        probs = probs[:, 1]
                    score = float(compute_roc_auc(y_val, probs))
                elif self.scoring == 'f1':
                    preds = model.predict(X_val, threshold=self.threshold)
                    score = float(compute_f1_score(y_val, preds))
                elif self.scoring == 'accuracy':
                    preds = model.predict(X_val, threshold=self.threshold)
                    score = float(compute_accuracy(y_val, preds))
                elif self.scoring == 'precision':
                    preds = model.predict(X_val, threshold=self.threshold)
                    score = float(compute_precision(y_val, preds))
                elif self.scoring == 'recall':
                    preds = model.predict(X_val, threshold=self.threshold)
                    score = float(compute_recall(y_val, preds))
                else:
                    raise ValueError(f"Chỉ số {self.scoring} không hợp lệ.")

                fold_scores.append(score)
                cv_results[f'split{fold_i}_test_score'][cand_i] = score

                if self.verbose >= 2:
                    print(f"     -> Fold {fold_i + 1}/{self.n_splits_}: {self.scoring}={score:.4f} ({fit_duration:.2f}s)")

            mean_sc = float(np.mean(fold_scores))
            std_sc = float(np.std(fold_scores))
            mean_time = float(np.mean(fold_fit_times))

            cv_results['mean_test_score'][cand_i] = mean_sc
            cv_results['std_test_score'][cand_i] = std_sc
            cv_results['mean_fit_time'][cand_i] = mean_time

            if self.verbose >= 1:
                print(f"     => Diem TB: {mean_sc:.4f} (+/- {std_sc:.4f}) | Thoi gian: {time.time()-t_cand_start:.2f}s")

        # Xếp hạng ứng viên (thứ tự giảm dần của mean_test_score)
        ranks = np.zeros(n_candidates, dtype=int)
        sorted_indices = np.argsort(-cv_results['mean_test_score'])
        for rank_pos, idx in enumerate(sorted_indices):
            ranks[idx] = rank_pos + 1
        cv_results['rank_test_score'] = ranks

        self.cv_results_ = cv_results
        self.best_index_ = int(sorted_indices[0])
        self.best_params_ = candidates[self.best_index_]
        self.best_score_ = float(cv_results['mean_test_score'][self.best_index_])

        total_time = time.time() - start_total
        self.total_search_time_ = float(total_time)
        self.refit_time_ = 0.0

        if self.verbose >= 1:
            print("-" * 78)
            print(f"[*] HOAN TAT TIM KIEM LUOI TRONG {total_time:.2f} GIAY")
            print(f"    - To hop tham so toi uu (Best Params): {self.best_params_}")
            print(f"    - Diem so tot nhat ({self.scoring})       : {self.best_score_:.4f} (+/- {cv_results['std_test_score'][self.best_index_]:.4f})")
            print("-" * 78)

        # Huấn luyện lại mô hình tốt nhất trên toàn bộ tập dữ liệu (refit)
        if self.refit:
            if self.verbose >= 1:
                print(f"[*] Dang huan luyen lai (refit) best_estimator_ tren toan bo du lieu {X_arr.shape[0]:,} mau...")
            t_refit_start = time.time()
            best_model = copy.deepcopy(self.estimator)
            best_model.set_params(**self.best_params_)
            best_model.fit(X_arr, y_arr, verbose=(self.verbose >= 2))
            self.refit_time_ = float(time.time() - t_refit_start)
            self.best_estimator_ = best_model
            if self.verbose >= 1:
                print(f"[*] Huan luyen lai hoan tat trong {self.refit_time_:.2f}s! Mo hinh da san sang du doan.\n")

        return self

    def predict(self, X, threshold=None):
        """Dự đoán nhãn nhị phân bằng best_estimator_."""
        if self.best_estimator_ is None:
            raise RuntimeError("Mô hình chưa được fit hoặc refit=False. Vui lòng fit với refit=True trước.")
        th = self.threshold if threshold is None else threshold
        return self.best_estimator_.predict(X, threshold=th)

    def predict_proba(self, X):
        """Dự đoán xác suất 1 chiều bằng best_estimator_."""
        if self.best_estimator_ is None:
            raise RuntimeError("Mô hình chưa được fit hoặc refit=False. Vui lòng fit với refit=True trước.")
        return self.best_estimator_.predict_proba(X)

    def predict_proba_2d(self, X):
        """Dự đoán xác suất 2 chiều [P(0), P(1)] bằng best_estimator_."""
        if self.best_estimator_ is None:
            raise RuntimeError("Mô hình chưa được fit hoặc refit=False. Vui lòng fit với refit=True trước.")
        return self.best_estimator_.predict_proba_2d(X)

    def predict_raw(self, X):
        """Dự đoán logit thô bằng best_estimator_."""
        if self.best_estimator_ is None:
            raise RuntimeError("Mô hình chưa được fit hoặc refit=False.")
        return self.best_estimator_.predict_raw(X)

    def score(self, X, y):
        """Trả về điểm số đánh giá trên tập dữ liệu kiểm tra mới."""
        if self.best_estimator_ is None:
            raise RuntimeError("Mô hình chưa được fit.")
        if self.scoring == 'roc_auc':
            return compute_roc_auc(y, self.predict_proba(X))
        elif self.scoring == 'f1':
            return compute_f1_score(y, self.predict(X, threshold=self.threshold))
        elif self.scoring == 'accuracy':
            return compute_accuracy(y, self.predict(X, threshold=self.threshold))
        elif self.scoring == 'precision':
            return compute_precision(y, self.predict(X, threshold=self.threshold))
        elif self.scoring == 'recall':
            return compute_recall(y, self.predict(X, threshold=self.threshold))
        else:
            return compute_accuracy(y, self.predict(X, threshold=self.threshold))

    def summary(self):
        """Trả về chuỗi bảng tổng hợp kết quả GridSearchCV."""
        if not self.cv_results_:
            return "Chua co ket qua. Vui long goi fit(X, y) truoc."
        lines = []
        lines.append(f"{'Hang':<5} | {'Diem TB':<10} | {'Do lech (Std)':<14} | {'Thoi gian':<10} | Thong so")
        lines.append("-" * 75)
        order = np.argsort(self.cv_results_['rank_test_score'])
        for idx in order:
            rank = self.cv_results_['rank_test_score'][idx]
            mean_sc = self.cv_results_['mean_test_score'][idx]
            std_sc = self.cv_results_['std_test_score'][idx]
            t_fit = self.cv_results_['mean_fit_time'][idx]
            params = self.cv_results_['params'][idx]
            p_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"{rank:<5} | {mean_sc:<10.4f} | {std_sc:<14.4f} | {t_fit:<9.2f}s | {p_str}")
        return "\n".join(lines)


__all__ = [
    'train_test_split_stratified',
    'compute_confusion_matrix',
    'compute_accuracy',
    'compute_precision',
    'compute_recall',
    'compute_f1_score',
    'compute_specificity',
    'compute_npv',
    'compute_roc_auc',
    'compute_roc_curve',
    'compute_precision_recall_curve',
    'roc_curve',
    'precision_recall_curve',
    'HistBinMapper',
    'HistTreeNode',
    'HistRegressionTree',
    'CustomHistGradientBoostingClassifier',
    'StratifiedKFold',
    'cross_val_score',
    'CustomGridSearchCV',
]


if __name__ == '__main__':
    print("=" * 78)
    print("  HISTOGRAM GRADIENT BOOSTING (HGB) -- COMPONENT VERIFICATION")
    print("  100% Pure Python & NumPy (Zero Scikit-Learn)")
    print("=" * 78)

    rng = np.random.RandomState(42)
    n_samples, n_features = 400, 10
    X_syn = rng.randn(n_samples, n_features).astype(np.float32)
    logit = 1.5 * X_syn[:, 0] - 2.0 * X_syn[:, 1] + 0.8 * (X_syn[:, 2] ** 2) - 0.5
    prob = 1.0 / (1.0 + np.exp(-logit))
    y_syn = (prob >= 0.5).astype(np.float32)

    print(f"[*] Synthetic dataset: {n_samples} samples, {n_features} features")
    print(f"    Class balance    : {int(y_syn.sum())} positive ({y_syn.mean()*100:.1f}%), {n_samples - int(y_syn.sum())} negative")

    # 1. Test train_test_split_stratified
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X_syn, y_syn, test_size=0.25, random_state=42)
    print(f"[v] Stratified Split : Train={len(y_tr)}, Test={len(y_te)} (distribution preserved)")

    # 2. Test HistBinMapper
    mapper = HistBinMapper(max_bins=64)
    mapper.fit(X_tr)
    X_tr_b = mapper.transform(X_tr)
    print(f"[v] HistBinMapper    : Successfully mapped to uint8 bins in [0, {X_tr_b.max()}]")

    # 3. Test CustomHistGradientBoostingClassifier
    clf = CustomHistGradientBoostingClassifier(
        n_estimators=30, learning_rate=0.1, max_depth=4,
        min_samples_leaf=10, l2_regularization=1.0, max_bins=64,
        validation_fraction=0.15, n_iter_no_change=10, random_state=42
    )
    clf.fit(X_tr, y_tr, verbose=False)
    print(f"[v] Classifier Fit   : Completed {clf.n_iter_} trees (early stopped / best iteration: {clf.best_n_iter_})")

    # 4. Test predictions and metrics
    y_pred = clf.predict(X_te, threshold=0.50)
    y_proba = clf.predict_proba(X_te)
    acc = compute_accuracy(y_te, y_pred)
    f1 = compute_f1_score(y_te, y_pred)
    auc = compute_roc_auc(y_te, y_proba)
    print(f"[v] Test Metrics     : Accuracy={acc*100:.2f}% | F1={f1*100:.2f}% | ROC-AUC={auc:.4f}")

    # 5. Test Cross-Validation
    cv_scores = cross_val_score(clf, X_tr, y_tr, cv=3, scoring='roc_auc')
    print(f"[v] 3-Fold Stratified: Mean ROC-AUC={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Test Curves
    fpr, tpr, _ = compute_roc_curve(y_te, y_proba)
    p_curve, r_curve, _ = compute_precision_recall_curve(y_te, y_proba)
    print(f"[v] ROC & PR Curves  : Evaluated ({len(fpr)} ROC points, {len(p_curve)} PR points)")

    print("=" * 78)
    print("  ALL HGB CORE MODULES OPERATIONAL & VERIFIED SUCCESSFULLY")
    print("=" * 78)

