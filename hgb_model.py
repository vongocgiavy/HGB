"""
Module hgb_model: Thuật toán Histogram Gradient Boosting (HGB) phân loại nhị phân,
triển khai 100% bằng Python thuần và NumPy (Zero Scikit-Learn).
Hỗ trợ đầy đủ: HistBinMapper, HistTreeNode, HistRegressionTree,
CustomHistGradientBoostingClassifier, StratifiedKFold, cross_val_score,
CustomGridSearchCV và bộ metrics đánh giá toàn diện.

Tối ưu hóa:
1. Vector hóa toàn bộ việc xây dựng Histogram trong HistRegressionTree._best_split
   qua chỉ số phẳng (flat index) và 3 lời gọi C-level bincount/cumsum theo axis=1.
2. Cố định n_bins = self.max_bins trên toàn bộ cây để không gian bin nhất quán.
3. Thêm n_bins_per_feature_ và check_bin_quality() trong HistBinMapper để minh bạch hóa độ phân giải bin.
4. Tách tầng seed độc lập nhưng tái lập (reproducible) cho _split_validation.
"""

import time
import copy
import itertools
import numpy as np


# ==============================================================================
# PHẦN 1: HÀM PHÂN CHIA DỮ LIỆU & CHỈ SỐ ĐÁNH GIÁ THUẦN NUMPY (ZERO SKLEARN)
# ==============================================================================

def train_test_split_stratified(X, y, test_size=0.2, random_state=42, return_indices=False):
    """
    Phân chia tập dữ liệu thành Train/Test phân tầng (Stratified),
    bảo toàn đúng tỷ lệ nhãn ở cả hai tập. Không dùng scikit-learn.

    Tham số:
        X              : DataFrame hoặc ndarray, ma trận đặc trưng.
        y              : ndarray, nhãn nhị phân {0, 1}.
        test_size      : float, tỷ lệ dành cho tập Test (mặc định 0.2).
        random_state   : int, hạt giống ngẫu nhiên (mặc định 42).
        return_indices : bool, nếu True trả về thêm (train_idx, test_idx).

    Trả về:
        Nếu return_indices=False:
            X_train, X_test, y_train, y_test (đều là ndarray)
        Nếu return_indices=True:
            X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size phai nam trong khoang (0.0, 1.0). Nhan duoc: {test_size}")

    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y).ravel()

    classes = np.unique(y_arr)
    if len(classes) < 2:
        raise ValueError(f"y phai chua it nhat 2 lop nhan. Nhan duoc: {classes.tolist()}")

    train_idx, test_idx = [], []
    for cls in classes:
        cls_indices = np.where(y_arr == cls)[0].copy()
        rng.shuffle(cls_indices)
        n_test = int(np.round(len(cls_indices) * test_size))
        test_idx.extend(cls_indices[:n_test])
        train_idx.extend(cls_indices[n_test:])

    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx  = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    if hasattr(X, 'iloc'):
        X_train = X.iloc[train_idx].values
        X_test  = X.iloc[test_idx].values
    else:
        X_arr = np.asarray(X)
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]

    if return_indices:
        return X_train, X_test, y_arr[train_idx], y_arr[test_idx], train_idx, test_idx
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
        raise ValueError(f"Kich thuoc khong khop: y_true ({len(y_t)}) != y_pred ({len(y_p)}).")
    if len(y_t) == 0:
        return 0, 0, 0, 0
    if not np.all(np.isin(y_t, [0, 1])):
        raise ValueError(f"y_true phai la nhan nhi phan {{0, 1}}. Nhan duoc: {np.unique(y_t).tolist()}")
    if not np.all(np.isin(y_p, [0, 1])):
        raise ValueError(f"y_pred phai la nhan nhi phan {{0, 1}}. Nhan duoc: {np.unique(y_p).tolist()}")

    idx = 2 * y_t.astype(np.int64) + y_p.astype(np.int64)
    counts = np.bincount(idx, minlength=4)
    return int(counts[3]), int(counts[0]), int(counts[1]), int(counts[2])


def compute_accuracy(y_true, y_pred):
    """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return float((tp + tn) / total) if total > 0 else 0.0


def compute_precision(y_true, y_pred):
    """Precision = TP / (TP + FP)"""
    tp, _, fp, _ = compute_confusion_matrix(y_true, y_pred)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def compute_recall(y_true, y_pred):
    """Recall (Sensitivity, TPR) = TP / (TP + FN)"""
    tp, _, _, fn = compute_confusion_matrix(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def compute_specificity(y_true, y_pred):
    """Specificity (True Negative Rate) = TN / (TN + FP)"""
    _, tn, fp, _ = compute_confusion_matrix(y_true, y_pred)
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def compute_npv(y_true, y_pred):
    """Negative Predictive Value = TN / (TN + FN)"""
    _, tn, _, fn = compute_confusion_matrix(y_true, y_pred)
    denom = tn + fn
    return float(tn / denom) if denom > 0 else 0.0


def compute_f1_score(y_true, y_pred):
    """F1-Score = 2 * Precision * Recall / (Precision + Recall)"""
    p = compute_precision(y_true, y_pred)
    r = compute_recall(y_true, y_pred)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_roc_auc(y_true, y_scores):
    """
    ROC-AUC theo công thức thống kê Mann-Whitney U / Wilcoxon rank-sum (Zero Sklearn).
    AUC = P(score(pos) > score(neg)).

    Tham số:
        y_true   : nhãn nhị phân {0, 1}
        y_scores : điểm số xác suất liên tục [0, 1]
    """
    y_true   = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if len(y_true) != len(y_scores):
        raise ValueError(f"Kich thuoc khong khop: y_true ({len(y_true)}) != y_scores ({len(y_scores)}).")
    if len(y_true) == 0:
        return 0.5
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true phai la nhan nhi phan {0, 1}.")
    if not np.isfinite(y_scores).all():
        raise ValueError("y_scores chua gia tri NaN hoac Inf.")

    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_scores) + 1, dtype=np.float64)

    # Xử lý tie (điểm bằng nhau): gán rank trung bình
    sorted_scores = y_scores[order]
    if np.any(sorted_scores[1:] == sorted_scores[:-1]):
        _, inv_idx, counts = np.unique(y_scores, return_inverse=True, return_counts=True)
        tie_ranks = np.cumsum(np.r_[0, counts[:-1]]) + (counts + 1) / 2.0
        ranks = tie_ranks[inv_idx]

    sum_pos_ranks = float(np.sum(ranks[pos_mask]))
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_roc_curve(y_true, y_scores, drop_intermediate: bool = True):
    """
    Tính False Positive Rate (FPR) và True Positive Rate (TPR) qua các ngưỡng phân loại (Zero Sklearn).

    Trả về:
        fpr, tpr, thresholds (np.ndarray)
    """
    y_t = np.asarray(y_true).ravel()
    scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if len(y_t) != len(scores):
        raise ValueError(f"Kich thuoc khong khop: y_true ({len(y_t)}) != y_scores ({len(scores)}).")
    if not np.all(np.isin(y_t, [0, 1])):
        raise ValueError("y_true phai la nhan nhi phan {0, 1}.")
    if not np.isfinite(scores).all():
        raise ValueError("y_scores chua gia tri NaN hoac Inf.")

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
    Tính Precision và Recall qua các ngưỡng phân loại tăng dần (Zero Sklearn).
    """
    y_t = np.asarray(y_true).ravel()
    scores = np.asarray(y_scores, dtype=np.float64).ravel()

    if len(y_t) != len(scores):
        raise ValueError(f"Kich thuoc khong khop: y_true ({len(y_t)}) != y_scores ({len(scores)}).")
    if not np.all(np.isin(y_t, [0, 1])):
        raise ValueError("y_true phai la nhan nhi phan {0, 1}.")
    if not np.isfinite(scores).all():
        raise ValueError("y_scores chua gia tri NaN hoac Inf.")

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

    p = np.r_[precision[::-1], 1.0]
    r = np.r_[recall[::-1], 0.0]
    th = scores_sorted[threshold_idxs][::-1]
    return p, r, th


# ==============================================================================
# PHẦN 2: HISTOGRAM BIN MAPPER
# ==============================================================================

class HistBinMapper:
    """
    Rời rạc hóa ma trận đặc trưng liên tục thành thùng số nguyên uint8
    bằng phương pháp phân vị đều (Quantile Binning).

    Lưu ý: Chỉ gọi fit() trên tập huấn luyện (fit-train hoặc full train tùy phase).
    Tuyệt đối không fit trên X_test.
    """

    def __init__(self, max_bins: int = 255):
        max_bins_int = int(max_bins)
        if not (2 <= max_bins_int <= 256):
            raise ValueError(
                f"max_bins phai nam trong khoang [2, 256] do bieu dien uint8 [0..255]. "
                f"Nhan duoc: {max_bins}"
            )
        self.max_bins = max_bins_int
        self.bin_thresholds_ = []
        self.n_bins_per_feature_ = []  # Lưu số bin thực tế cho từng đặc trưng
        self.n_features_in_ = None

    def fit(self, X) -> "HistBinMapper":
        """
        Tính ngưỡng phân vị đều cho mỗi đặc trưng trên dữ liệu train.
        """
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("Du lieu X chua gia tri khong hop le (NaN hoac +/-Inf).")

        self.n_features_in_ = X.shape[1]
        self.bin_thresholds_ = []
        self.n_bins_per_feature_ = []
        percentiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]

        for j in range(self.n_features_in_):
            thresholds = np.nanpercentile(X[:, j], percentiles)
            thresholds = np.unique(thresholds)
            self.bin_thresholds_.append(thresholds)
            # Với T ngưỡng, số bin thực tế ánh xạ được là T + 1
            self.n_bins_per_feature_.append(int(len(thresholds) + 1))
        return self

    def check_bin_quality(self, min_ratio: float = 0.5, verbose: bool = True) -> list:
        """
        Kiểm tra độ phân giải của các bin cho từng đặc trưng.
        Cảnh báo nếu số bin thực tế < min_ratio * max_bins (ví dụ < 50% max_bins).

        Trả về:
            list các tuple: (feature_idx, actual_bins, max_bins, ratio)
        """
        if not self.n_bins_per_feature_:
            raise RuntimeError("HistBinMapper chua duoc fit. Vui long goi fit() truoc.")

        warnings = []
        threshold_bins = self.max_bins * min_ratio
        for j, actual_bins in enumerate(self.n_bins_per_feature_):
            if actual_bins < threshold_bins:
                ratio = actual_bins / self.max_bins
                warnings.append((j, actual_bins, self.max_bins, ratio))
                if verbose:
                    print(f"[CẢNH BÁO Binning] Đặc trưng {j}: số bin thực tế ({actual_bins}) "
                          f"< {min_ratio*100:.0f}% max_bins ({self.max_bins}) - tỷ lệ: {ratio*100:.1f}%. "
                          f"Đặc trưng có thể có nhiều giá trị trùng lặp hoặc phân phối rời rạc.")
        return warnings

    def transform(self, X) -> np.ndarray:
        """
        Ánh xạ X thành ma trận uint8 chỉ số bin.
        """
        if self.n_features_in_ is None or len(self.bin_thresholds_) == 0:
            raise RuntimeError("HistBinMapper chua duoc fit. Vui long goi fit() truoc khi transform().")

        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"So dac trung khong khop: fit voi {self.n_features_in_} cot, "
                f"nhung transform nhan {X.shape[1]} cot."
            )
        if not np.isfinite(X).all():
            raise ValueError("Du lieu X truyen vao transform chua gia tri NaN hoac +/-Inf.")

        n_samples, n_features = X.shape
        X_binned = np.empty((n_samples, n_features), dtype=np.uint8)

        for j in range(n_features):
            X_binned[:, j] = np.searchsorted(
                self.bin_thresholds_[j], X[:, j], side='right'
            ).astype(np.uint8)
        return X_binned


# ==============================================================================
# PHẦN 3: CÂY QUYẾT ĐỊNH HISTOGRAM
# ==============================================================================

class HistTreeNode:
    """Nút cây quyết định Histogram."""
    __slots__ = ('is_leaf', 'value', 'feature_idx', 'bin_threshold', 'gain', 'left', 'right')

    def __init__(self, is_leaf=False, value=0.0, feature_idx=None,
                 bin_threshold=None, gain=0.0):
        self.is_leaf       = is_leaf
        self.value         = float(value)
        self.feature_idx   = feature_idx
        self.bin_threshold = bin_threshold
        self.gain          = float(gain)
        self.left          = None
        self.right         = None


class HistRegressionTree:
    """
    Cây hồi quy Histogram tối ưu hóa dựa trên Gradient (g) và Hessian (h).
    Tìm điểm chia O(D * K) qua np.bincount và np.cumsum vector hóa toàn bộ đặc trưng.
    """

    def __init__(self, max_depth=6, min_samples_leaf=20,
                 l2_regularization=1.0, min_gain_to_split=1e-7, max_bins=255):
        self.max_depth        = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_reg           = float(l2_regularization)
        self.min_gain         = float(min_gain_to_split)
        self.max_bins         = int(max_bins)
        self.root             = None

    def fit(self, X_binned: np.ndarray, g: np.ndarray, h: np.ndarray) -> "HistRegressionTree":
        indices = np.arange(len(g))
        self.root = self._build(X_binned, g, h, indices, depth=0)
        return self

    def _leaf_weight(self, g, h, idx):
        if len(idx) == 0:
            return 0.0
        g_sum = np.sum(g[idx])
        h_sum = np.sum(h[idx])
        return float(-g_sum / (h_sum + self.l2_reg))

    def _best_split(self, X_binned, g, h, idx):
        """
        Tìm điểm chia tối ưu nhất trên toàn bộ đặc trưng bằng Histogram vector hóa.
        Độ phức tạp O(D * K), sử dụng 3 lời gọi bincount C-level duy nhất trên chỉ số phẳng.
        Không gian bin cố định n_bins = self.max_bins đảm bảo tính nhất quán toàn cục.
        """
        g_node = g[idx]
        h_node = h[idx]
        G_tot = float(np.sum(g_node))
        H_tot = float(np.sum(h_node))
        score_tot = (G_tot ** 2) / (H_tot + self.l2_reg)

        n_samples = len(idx)
        n_features = X_binned.shape[1]
        n_bins = self.max_bins

        # Vector hóa: Tính chỉ số phẳng flat_idx = feat * n_bins + bin cho tất cả đặc trưng
        feat_offsets = np.arange(n_features, dtype=np.int64) * n_bins
        flat_bins = (X_binned[idx].astype(np.int64) + feat_offsets).ravel()
        total_len = n_features * n_bins

        # Lặp lại gradient và hessian n_features lần (tương ứng từng mẫu)
        g_rep = np.repeat(g_node, n_features)
        h_rep = np.repeat(h_node, n_features)

        # 3 lời gọi C-level bincount cho toàn bộ D đặc trưng cùng lúc
        G_all = np.bincount(flat_bins, weights=g_rep, minlength=total_len).reshape(n_features, n_bins)
        H_all = np.bincount(flat_bins, weights=h_rep, minlength=total_len).reshape(n_features, n_bins)
        N_all = np.bincount(flat_bins, minlength=total_len).reshape(n_features, n_bins)

        # Tính tổng tích lũy bên trái dọc theo trục bin (axis=1)
        G_L = np.cumsum(G_all, axis=1)
        H_L = np.cumsum(H_all, axis=1)
        N_L = np.cumsum(N_all, axis=1)

        G_R = G_tot - G_L
        H_R = H_tot - H_L
        N_R = n_samples - N_L

        # Kiểm tra điều kiện tách hợp lệ: cả 2 nhánh con đều phải >= min_samples_leaf
        valid = (N_L >= self.min_samples_leaf) & (N_R >= self.min_samples_leaf)
        if not np.any(valid):
            return None

        # Tính điểm score cho tất cả các điểm cắt hợp lệ
        scores = np.full((n_features, n_bins), -np.inf, dtype=np.float64)
        scores[valid] = (G_L[valid] ** 2) / (H_L[valid] + self.l2_reg) + \
                        (G_R[valid] ** 2) / (H_R[valid] + self.l2_reg)

        max_flat_idx = int(np.argmax(scores))
        max_score = float(scores.ravel()[max_flat_idx])

        if np.isneginf(max_score):
            return None

        best_gain = float(0.5 * (max_score - score_tot))
        if best_gain > self.min_gain:
            best_feat = int(max_flat_idx // n_bins)
            best_bin  = int(max_flat_idx % n_bins)
            return best_feat, best_bin, best_gain

        return None

    def _build(self, X_binned, g, h, idx, depth):
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return HistTreeNode(is_leaf=True, value=self._leaf_weight(g, h, idx))

        split = self._best_split(X_binned, g, h, idx)
        if split is None:
            return HistTreeNode(is_leaf=True, value=self._leaf_weight(g, h, idx))

        feat, bin_thr, gain = split
        mask = X_binned[idx, feat] <= bin_thr
        left_idx = idx[mask]
        right_idx = idx[~mask]

        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return HistTreeNode(is_leaf=True, value=self._leaf_weight(g, h, idx))

        node = HistTreeNode(is_leaf=False, feature_idx=feat,
                            bin_threshold=bin_thr, gain=gain)
        node.left  = self._build(X_binned, g, h, left_idx,  depth + 1)
        node.right = self._build(X_binned, g, h, right_idx, depth + 1)
        return node

    def predict(self, X_binned: np.ndarray) -> np.ndarray:
        preds = np.zeros(X_binned.shape[0], dtype=np.float32)
        self._traverse(self.root, X_binned, np.arange(len(preds)), preds)
        return preds

    def _traverse(self, node, X_binned, idx, preds):
        if len(idx) == 0:
            return
        if node.is_leaf:
            preds[idx] = node.value
            return
        mask = X_binned[idx, node.feature_idx] <= node.bin_threshold
        self._traverse(node.left,  X_binned, idx[mask],  preds)
        self._traverse(node.right, X_binned, idx[~mask], preds)

    def compute_feature_importances(self, n_features: int) -> np.ndarray:
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
    Bộ phân loại Histogram Gradient Boosting (HGB) thuần NumPy - Zero Scikit-Learn.
    
    Hỗ trợ 2 chế độ:
      - Development Mode: validation_fraction > 0 -> tách validation nội bộ, chạy Early Stopping.
      - Production / Full Fit Mode: validation_fraction = 0.0 -> fit toàn bộ X mà không tách validation.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        min_gain_to_split=1e-7,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
        random_state=42
    ):
        self.n_estimators        = int(n_estimators)
        self.learning_rate       = float(learning_rate)
        self.max_depth           = int(max_depth)
        self.min_samples_leaf    = int(min_samples_leaf)
        self.l2_reg              = float(l2_regularization)
        self.l2_regularization   = float(l2_regularization)
        self.max_bins            = int(max_bins)
        self.min_gain            = float(min_gain_to_split)
        self.min_gain_to_split   = float(min_gain_to_split)
        self.validation_fraction = float(validation_fraction) if validation_fraction is not None else 0.0
        self.n_iter_no_change    = int(n_iter_no_change) if n_iter_no_change is not None else 0
        self.tol                 = float(tol)
        self.random_state        = random_state

        self._validate_params()

        # Trạng thái mô hình
        self.bin_mapper          = None
        self.trees               = []
        self.base_score_         = 0.0
        self.n_features_in_      = None
        self.classes_            = np.array([0, 1], dtype=np.int32)
        self.n_iter_             = 0
        self.best_n_iter_        = 0
        self.stopped_iter_       = 0
        self.best_val_loss_      = np.inf
        self.feature_importances_ = None
        self.train_loss_history_       = []
        self.val_loss_history_         = []
        self.full_train_loss_history_  = []
        self.full_val_loss_history_    = []
        self.fit_time_                 = 0.0

        # Lưu validation subsets tách nội bộ
        self.X_train_sub_ = None
        self.y_train_sub_ = None
        self.X_val_       = None
        self.y_val_       = None

    def _validate_params(self):
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
        if not (0.0 <= self.validation_fraction < 1.0):
            raise ValueError(f"validation_fraction phai nam trong khoang [0.0, 1.0) (nhan duoc: {self.validation_fraction}).")
        if self.validation_fraction > 0 and self.n_iter_no_change <= 0:
            raise ValueError(f"n_iter_no_change (patience) phai > 0 khi co validation (nhan duoc: {self.n_iter_no_change}).")
        if self.tol < 0:
            raise ValueError(f"tol phai >= 0 (nhan duoc: {self.tol}).")

    def get_params(self, deep=True):
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
        for key, value in params.items():
            if key in ('l2_regularization', 'l2_reg'):
                self.l2_reg = float(value)
                self.l2_regularization = float(value)
            elif key in ('min_gain_to_split', 'min_gain'):
                self.min_gain = float(value)
                self.min_gain_to_split = float(value)
            elif key in ('n_estimators', 'max_depth', 'min_samples_leaf', 'n_iter_no_change'):
                setattr(self, key, int(value) if value is not None else 0)
            elif key in ('learning_rate', 'validation_fraction', 'tol'):
                setattr(self, key, float(value))
            elif key == 'max_bins':
                self.max_bins = int(value)
            else:
                setattr(self, key, value)
        self._validate_params()
        return self

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    @staticmethod
    def _log_loss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
        p = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def _split_validation(self, X, y):
        """
        Tách Validation nội bộ phân tầng (stratified) một cách độc lập.
        Sử dụng seed dẫn xuất xác định từ self.random_state để đảm bảo tính độc lập
        ngẫu nhiên với tầng chia Train/Test bên ngoài nhưng vẫn hoàn toàn tái lập.
        """
        if isinstance(self.random_state, (int, np.integer)):
            # Dẫn xuất seed riêng cho validation bằng công thức xác định (deterministic hash)
            val_seed = int((int(self.random_state) * 1664525 + 1013904223) % (2**31 - 1))
            rng = np.random.RandomState(val_seed)
        elif isinstance(self.random_state, np.random.RandomState):
            val_seed = self.random_state.randint(0, 2**31 - 1)
            rng = np.random.RandomState(val_seed)
        else:
            rng = np.random.RandomState(10007)

        val_idx, tr_idx = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0].copy()
            rng.shuffle(idx)
            n_val = max(1, int(np.round(len(idx) * self.validation_fraction)))
            val_idx.extend(idx[:n_val])
            tr_idx.extend(idx[n_val:])
        tr_idx  = np.array(tr_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    def fit(self, X, y, verbose=False) -> "CustomHistGradientBoostingClassifier":
        """
        Huấn luyện mô hình HGB.
        - Nếu validation_fraction > 0: tách validation set nội bộ để giám sát Early Stopping.
        - Nếu validation_fraction == 0: huấn luyện trên 100% dữ liệu X truyền vào.
        """
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        self._validate_params()

        if not np.isfinite(X).all():
            raise ValueError("Du lieu X chua gia tri khong hop le (NaN hoac +/-Inf).")
        if not np.isfinite(y).all():
            raise ValueError("Nhan y chua gia tri khong hop le (NaN hoac +/-Inf).")

        unique_y = np.unique(y)
        if not np.all(np.isin(unique_y, [0, 1])):
            raise ValueError(f"CustomHistGradientBoostingClassifier chi ho tro nhan {{0, 1}}. Nhan: {unique_y.tolist()}")

        self.n_features_in_ = X.shape[1]

        # Reset hoàn toàn trạng thái giữa các lần fit
        self.trees               = []
        self.train_loss_history_ = []
        self.val_loss_history_   = []
        self.full_train_loss_history_ = []
        self.full_val_loss_history_   = []
        self.best_val_loss_      = np.inf
        self.best_n_iter_        = 0
        self.stopped_iter_       = 0
        no_improve               = 0

        use_val = (self.validation_fraction > 0.0) and (self.n_iter_no_change > 0)

        if use_val:
            X_raw_tr, y_tr, X_raw_val, y_val = self._split_validation(X, y)
            self.X_train_sub_ = X_raw_tr
            self.y_train_sub_ = y_tr
            self.X_val_       = X_raw_val
            self.y_val_       = y_val
        else:
            X_raw_tr = X
            y_tr = y
            X_raw_val = None
            y_val = None
            self.X_train_sub_ = X
            self.y_train_sub_ = y
            self.X_val_       = None
            self.y_val_       = None

        # Rời rạc hóa: fit bin mapper CHỈ trên X_raw_tr
        self.bin_mapper = HistBinMapper(max_bins=self.max_bins)
        self.bin_mapper.fit(X_raw_tr)
        X_tr = self.bin_mapper.transform(X_raw_tr)
        if use_val:
            X_val = self.bin_mapper.transform(X_raw_val)

        # F_0 = log-odds
        y_mean = float(np.clip(np.mean(y_tr), 1e-7, 1.0 - 1e-7))
        self.base_score_ = float(np.log(y_mean / (1.0 - y_mean)))
        F_tr = np.full(len(y_tr), self.base_score_, dtype=np.float32)
        if use_val:
            F_val = np.full(len(y_val), self.base_score_, dtype=np.float32)

        if verbose:
            print(f"[*] Bat dau huan luyen HGB ({self.n_estimators} cay toi da, lr={self.learning_rate})...")

        t0 = time.time()
        for m in range(1, self.n_estimators + 1):
            p_tr = self._sigmoid(F_tr)
            g = p_tr - y_tr
            h = np.maximum(p_tr * (1.0 - p_tr), 1e-16)

            tree = HistRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_reg,
                min_gain_to_split=self.min_gain,
                max_bins=self.max_bins,
            )
            tree.fit(X_tr, g, h)
            self.trees.append(tree)

            F_tr += self.learning_rate * tree.predict(X_tr)
            tl = self._log_loss(y_tr, self._sigmoid(F_tr))
            self.train_loss_history_.append(tl)

            if use_val:
                F_val += self.learning_rate * tree.predict(X_val)
                vl = self._log_loss(y_val, self._sigmoid(F_val))
                self.val_loss_history_.append(vl)

                improved = vl < (self.best_val_loss_ - self.tol)
                if improved:
                    self.best_val_loss_ = vl
                    self.best_n_iter_   = m
                    no_improve          = 0
                else:
                    no_improve += 1

                if no_improve >= self.n_iter_no_change:
                    if verbose:
                        print(f"[*] Early stopping triggered tai vong {m} (best={self.best_n_iter_}).")
                    break
            else:
                self.best_n_iter_ = m

        self.stopped_iter_ = len(self.trees)
        self.full_train_loss_history_ = list(self.train_loss_history_)
        self.full_val_loss_history_   = list(self.val_loss_history_)

        if use_val and self.best_n_iter_ > 0:
            self.trees = self.trees[:self.best_n_iter_]
            self.train_loss_history_ = self.train_loss_history_[:self.best_n_iter_]
            self.val_loss_history_   = self.val_loss_history_[:self.best_n_iter_]
        self.n_iter_ = len(self.trees)
        self.fit_time_ = float(time.time() - t0)

        # Tính Feature Importances (Gain-based)
        raw_importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree in self.trees:
            raw_importances += tree.compute_feature_importances(self.n_features_in_)
        tot_gain = np.sum(raw_importances)
        if tot_gain > 0:
            self.feature_importances_ = (raw_importances / tot_gain).astype(np.float64)
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)

        return self

    def _check_is_fitted(self):
        if self.bin_mapper is None or len(self.trees) == 0 or self.n_features_in_ is None:
            raise RuntimeError("Mo hinh chua duoc fit. Vui long goi fit(X, y) truoc khi du doan.")

    def predict_raw(self, X) -> np.ndarray:
        self._check_is_fitted()
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"So dac trung khong khop: fit {self.n_features_in_}, predict {X.shape[1]}.")
        if not np.isfinite(X).all():
            raise ValueError("Du lieu X truyen vao predict chua NaN hoac Inf.")

        X_b = self.bin_mapper.transform(X)
        F = np.full(X_b.shape[0], self.base_score_, dtype=np.float32)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X_b)
        return F

    def predict_proba(self, X) -> np.ndarray:
        return self._sigmoid(self.predict_raw(X))

    def predict_proba_2d(self, X) -> np.ndarray:
        p1 = self.predict_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: float = 0.50) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y) -> float:
        preds = self.predict(X, threshold=0.50)
        return compute_accuracy(y, preds)


# ==============================================================================
# PHẦN 5: KIỂM TRA CHÉO (CROSS-VALIDATION) & GRID SEARCH CV (ZERO SKLEARN)
# ==============================================================================

class StratifiedKFold:
    """K-Fold phân tầng thuần NumPy (Zero Scikit-Learn)."""

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        if n_splits < 2:
            raise ValueError(f"n_splits phai >= 2, nhan {n_splits}.")
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(self, X, y):
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
        return self.n_splits


def cross_val_score(estimator, X, y, cv=5, scoring='roc_auc', threshold=0.50, verbose=0):
    if isinstance(cv, int):
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                      random_state=getattr(estimator, 'random_state', 42))
    else:
        cv_splitter = cv

    if hasattr(X, 'iloc'):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    scores = []
    for train_idx, val_idx in cv_splitter.split(X_arr, y_arr):
        model = copy.deepcopy(estimator)
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]

        model.fit(X_tr, y_tr, verbose=False)

        if callable(scoring):
            score = float(scoring(model, X_val, y_val))
        elif scoring == 'roc_auc':
            probs = model.predict_proba(X_val)
            score = float(compute_roc_auc(y_val, probs))
        elif scoring == 'f1':
            preds = model.predict(X_val, threshold=threshold)
            score = float(compute_f1_score(y_val, preds))
        elif scoring == 'accuracy':
            preds = model.predict(X_val, threshold=threshold)
            score = float(compute_accuracy(y_val, preds))
        else:
            raise ValueError(f"Scoring '{scoring}' khong duoc ho tro.")
        scores.append(score)

    return np.array(scores, dtype=np.float64)


class CustomGridSearchCV:
    """Tìm kiếm lưới siêu tham số K-Fold phân tầng (Zero Scikit-Learn)."""

    def __init__(
        self,
        estimator,
        param_grid,
        scoring='roc_auc',
        cv=5,
        refit=True,
        threshold=0.50,
        verbose=1
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.threshold = threshold
        self.verbose = verbose

        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        self.best_index_ = -1
        self.cv_results_ = {}
        self.n_splits_ = 0

    def _generate_param_candidates(self):
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
            print(f"    - To hop tham so : {n_candidates}")
            print(f"    - Folds          : {self.n_splits_} (Stratified K-Fold)")
            print(f"    - Tieu chi       : {self.scoring.upper()}")
            print("=" * 78)

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

            for fold_i, (train_idx, val_idx) in enumerate(cv_splitter.split(X_arr, y_arr)):
                t0 = time.time()
                model = copy.deepcopy(self.estimator)
                model.set_params(**params)

                X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
                X_val, y_val = X_arr[val_idx], y_arr[val_idx]

                model.fit(X_tr, y_tr, verbose=False)
                fit_duration = time.time() - t0
                fold_fit_times.append(fit_duration)

                if callable(self.scoring):
                    score = float(self.scoring(model, X_val, y_val))
                elif self.scoring == 'roc_auc':
                    probs = model.predict_proba(X_val)
                    score = float(compute_roc_auc(y_val, probs))
                elif self.scoring == 'f1':
                    preds = model.predict(X_val, threshold=self.threshold)
                    score = float(compute_f1_score(y_val, preds))
                elif self.scoring == 'accuracy':
                    preds = model.predict(X_val, threshold=self.threshold)
                    score = float(compute_accuracy(y_val, preds))
                else:
                    raise ValueError(f"Chi so {self.scoring} khong hop le.")

                fold_scores.append(score)
                cv_results[f'split{fold_i}_test_score'][cand_i] = score

            mean_sc = float(np.mean(fold_scores))
            std_sc = float(np.std(fold_scores))
            mean_time = float(np.mean(fold_fit_times))

            cv_results['mean_test_score'][cand_i] = mean_sc
            cv_results['std_test_score'][cand_i] = std_sc
            cv_results['mean_fit_time'][cand_i] = mean_time

            if self.verbose >= 1:
                p_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"[{cand_i + 1:02d}/{n_candidates:02d}] {{{p_str}}} => Score: {mean_sc:.4f} (+/- {std_sc:.4f})")

        sorted_indices = np.argsort(-cv_results['mean_test_score'])
        ranks = np.zeros(n_candidates, dtype=int)
        for rank_pos, idx in enumerate(sorted_indices):
            ranks[idx] = rank_pos + 1
        cv_results['rank_test_score'] = ranks

        self.cv_results_ = cv_results
        self.best_index_ = int(sorted_indices[0])
        self.best_params_ = candidates[self.best_index_]
        self.best_score_ = float(cv_results['mean_test_score'][self.best_index_])
        self.total_search_time_ = float(time.time() - start_total)

        if self.refit:
            if self.verbose >= 1:
                print(f"[*] Refit best estimator tren toan bo {X_arr.shape[0]:,} mau...")
            best_model = copy.deepcopy(self.estimator)
            best_model.set_params(**self.best_params_)
            best_model.fit(X_arr, y_arr, verbose=(self.verbose >= 1))
            self.best_estimator_ = best_model

        return self

    def predict(self, X, threshold=None):
        if self.best_estimator_ is None:
            raise RuntimeError("Mo hinh chua duoc fit hoac refit=False.")
        th = self.threshold if threshold is None else threshold
        return self.best_estimator_.predict(X, threshold=th)

    def predict_proba(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Mo hinh chua duoc fit hoac refit=False.")
        return self.best_estimator_.predict_proba(X)


__all__ = [
    'train_test_split_stratified',
    'compute_confusion_matrix',
    'compute_accuracy',
    'compute_precision',
    'compute_recall',
    'compute_specificity',
    'compute_npv',
    'compute_f1_score',
    'compute_roc_auc',
    'compute_roc_curve',
    'compute_precision_recall_curve',
    'HistBinMapper',
    'HistTreeNode',
    'HistRegressionTree',
    'CustomHistGradientBoostingClassifier',
    'StratifiedKFold',
    'cross_val_score',
    'CustomGridSearchCV',
]
