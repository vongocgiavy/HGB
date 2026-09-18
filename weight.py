"""
weight.py — Tập trung toàn bộ siêu tham số, trọng số và cấu hình pipeline
cho dự án Histogram Gradient Boosting (HGB) phân loại SUSY.

"""

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA ĐẶC TRƯNG (SUSY Dataset — Baldi et al. 2014)
# UCI doi:10.24432/C54606
# Col 0 = label (1 = SUSY signal, 0 = SM background)
# Col 1–8  = 8 đặc trưng động học cấp thấp (low-level)
# Col 9–18 = 10 đặc trưng dẫn xuất cấp cao (high-level)
# ==============================================================================

FEATURE_NAMES = [
    "lepton1_pT",    "lepton1_eta",   "lepton1_phi",
    "lepton2_pT",    "lepton2_eta",   "lepton2_phi",
    "MET_magnitude", "MET_phi",
    "MET_rel",       "axial_MET",     "M_R",      "M_TR_2",
    "R",             "MT2",           "S_R",
    "M_Delta_R",     "dPhi_r_b",      "cos_theta_r1",
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

# Tập đặc trưng low-level (dùng để phân loại trong báo cáo)
LOW_LEVEL_FEATURES = {
    "lepton1_pT", "lepton1_eta", "lepton1_phi",
    "lepton2_pT", "lepton2_eta", "lepton2_phi",
    "MET_magnitude", "MET_phi",
}


# ==============================================================================
# PHẦN 2: CẤU HÌNH DỮ LIỆU & PHÂN CHIA
# ==============================================================================

DATA_CONFIG = {
    # Đường dẫn tương đối tới file CSV (tính từ thư mục dự án)
    "data_subdir":  "data",
    "filename":     "SUSY.csv",

    # Tỷ lệ phân chia Train / Test (stratified)
    "test_size":    0.2,      # 20% Test, 80% Train

    # Số mẫu tối đa khi chạy full (None = không giới hạn)
    "full_nrows":   None,

    # Kiểm tra cứng kích thước khi chạy full dataset
    "expected_full_samples":  5_000_000,
    "expected_train_samples": 4_000_000,
    "expected_test_samples":  1_000_000,
}


# ==============================================================================
# PHẦN 3: SIÊU THAM SỐ MÔ HÌNH HGB (Base Hyperparameters)
# Đây là bộ tham số mặc định dùng khi KHÔNG chạy Grid Search.
#
# ★ THAM SỐ TỐI ƯU (cập nhật từ lần chạy outputs/ — 60,000 mẫu, random_state=42)
#   - n_estimators = 74   ← best_iteration từ Early Stopping (Phase 1 Dev)
#   - learning_rate = 0.1 ← giữ nguyên (không đổi qua Grid Search)
#   - max_depth    = 6    ← giữ nguyên
#   - min_samples_leaf = 20  ← giữ nguyên
#   - l2_regularization = 1.0 ← giữ nguyên
#   Kết quả Test (12,000 mẫu): Accuracy=79.26% | F1=77.89% | ROC-AUC=0.8767
# ==============================================================================

HGB_BASE_PARAMS = {
    # ── Cấu trúc rừng cây ─────────────────────────────────────────────────────
    "n_estimators":      74,     # ★ TỐI ƯU: best_iteration từ Early Stopping (Phase 1, 60k mẫu)
                                 #   (Giá trị gốc: 200 — Early Stopping sẽ dừng sớm nếu cần)
    "max_depth":         6,      # Độ sâu tối đa mỗi cây

    # ── Tốc độ học ────────────────────────────────────────────────────────────
    "learning_rate":     0.1,    # Shrinkage factor (eta); nhỏ hơn → ổn định hơn, cần nhiều cây hơn

    # ── Chính quy hóa ─────────────────────────────────────────────────────────
    "l2_regularization": 1.0,    # Hệ số L2 (lambda) trên trọng số lá; tăng để giảm overfitting
    "min_samples_leaf":  20,     # Số mẫu tối thiểu tại nút lá; tăng để cây "thô" hơn
    "min_gain_to_split": 1e-3,   # Gain tối thiểu để thực hiện tách nút; tăng để cắt tỉa mạnh hơn

    # ── Lượng tử hóa bin ──────────────────────────────────────────────────────
    "max_bins":          255,    # Số bin tối đa mỗi đặc trưng [2, 256]; 255 = uint8 max

    # ── Seed tái lập ──────────────────────────────────────────────────────────
    "random_state":      42,
}


# ==============================================================================
# PHẦN 4: CẤU HÌNH EARLY STOPPING & VALIDATION NỘI BỘ (Phase 1 — Dev Model)
# ==============================================================================

EARLY_STOPPING_CONFIG = {
    "validation_fraction": 0.1,   # 10% của X_train tách ra làm validation nội bộ
    "n_iter_no_change":    20,     # Số vòng lặp không cải thiện trước khi dừng (patience)
    "tol":                 1e-4,   # Ngưỡng cải thiện tối thiểu để tính là "có cải thiện"
}


# ==============================================================================
# PHẦN 5: NGƯỠNG PHÂN LOẠI — THRESHOLD SWEEP
# Quét trên Validation Set để chọn ngưỡng tau* tối ưu theo F1-Score.
# Ngưỡng được KHÓA trước khi đánh giá Test Set (tránh data leakage).
# ==============================================================================

THRESHOLD_CONFIG = {
    # Danh sách ngưỡng quét (quét tuần tự, chọn ngưỡng có F1 Val cao nhất)
    "sweep_thresholds": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],

    # ★ TỐI ƯU: tau* = 0.40 — được chọn tự động theo max F1-Score trên Validation Set
    #   (lần chạy 60,000 mẫu, Val F1 cao nhất tại threshold=0.40)
    "default_threshold": 0.40,
}


# ==============================================================================
# PHẦN 6: CẤU HÌNH GRID SEARCH (tùy chọn — bật bằng --grid_search)
# Grid Search chạy trên SUBSET của X_train để tiết kiệm thời gian.
# Kết quả best_params sẽ ghi đè lên HGB_BASE_PARAMS.
# ==============================================================================

GRID_SEARCH_CONFIG = {
    # Kích thước subset tối đa để chạy Grid Search (min với train_samples)
    "subset_size": 60_000,

    # Số folds cross-validation trong Grid Search
    "cv_folds": 3,

    # Tiêu chí đánh giá trong Grid Search
    "scoring": "roc_auc",

    # Số cây rút gọn khi Grid Search (để tăng tốc tìm kiếm)
    "gs_n_estimators_cap": 50,  # min(HGB_BASE_PARAMS["n_estimators"], gs_n_estimators_cap)

    # Không gian tham số cần tìm kiếm
    "param_grid": {
        "learning_rate":     [0.08, 0.1],
        "max_depth":         [5, 6],
        "min_samples_leaf":  [20, 30],
        "l2_regularization": [0.1, 1.0],
    },
}


# ==============================================================================
# PHẦN 7: CẤU HÌNH ĐẦU RA (outputs/)
# ==============================================================================

OUTPUT_CONFIG = {
    "outputs_subdir": "outputs",

    # Tên các file đầu ra
    "metrics_json":                      "metrics.json",
    "confusion_matrix_csv":              "confusion_matrix.csv",
    "threshold_sweep_csv":               "threshold_sweep.csv",
    "feature_importance_gain_csv":       "feature_importance_gain.csv",
    "feature_importance_permutation_csv":"feature_importance_permutation.csv",
    "training_history_csv":              "training_history.csv",
    "config_json":                       "config.json",
    "environment_json":                  "environment.json",
    "evaluation_summary_txt":            "evaluation_summary.txt",
}


# ==============================================================================
# PHẦN 8: CẤU HÌNH BÁO CÁO (evaluation_summary.txt)
# ==============================================================================

REPORT_CONFIG = {
    "report_width":   88,   # Chiều rộng khung báo cáo (số ký tự)
    "bar_length":     12,   # Chiều dài thanh tiến trình trực quan hóa
    "top_n_features":  5,   # Số đặc trưng hiển thị trong Top-N của báo cáo
}
