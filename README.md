# Phân loại Hạt Siêu Đối Xứng (SUSY) bằng Histogram Gradient Boosting (HGB)

Dự án nghiên cứu và triển khai hoàn chỉnh thuật toán **Histogram Gradient Boosting (HGB)** từ đầu 100% bằng **Python thuần và NumPy** (Zero Scikit-Learn trong toàn bộ thuật toán cốt lõi và hệ thống metrics). Mô hình được tối ưu hóa chuyên sâu để phân loại các sự kiện va chạm hạt siêu đối xứng (Supersymmetric Particles) trên tập dữ liệu chuẩn quốc tế **SUSY Benchmark** từ **UCI Machine Learning Repository** (5,000,000 mẫu dữ liệu).

---

## Mục lục
1. [Tổng quan Bài toán & Dữ liệu](#1-tổng-quan-bài-toán--dữ-liệu)
2. [Cấu trúc Thư mục Dự án](#2-cấu-trúc-thư-mục-dự-án)
3. [Cơ sở Lý thuyết & Toán học của Thuật toán HGB](#3-cơ-sở-lý-thuyết--toán-học-của-thuật-toán-hgb)
4. [Quy trình Huấn luyện 2-Phase & Ngăn chặn Rò rỉ Dữ liệu](#4-quy-trình-huấn-luyện-2-phase--ngăn-chặn-rò-rỉ-dữ-liệu)
5. [Chi tiết Kiến trúc Các Lớp & Hàm (Zero Scikit-Learn)](#5-chi-tiết-kiến-trúc-các-lớp--hàm-zero-scikit-learn)
6. [Hướng dẫn Sử dụng Dòng lệnh (CLI) & Kiểm thử](#6-hướng-dẫn-sử-dụng-dòng-lệnh-cli--kiểm-thử)
7. [Kiểm toán Rò rỉ Dữ liệu (Data Leakage Audit 14 Điểm)](#7-kiểm-toán-rò-rỉ-dữ-liệu-data-leakage-audit-14-điểm)
8. [Kết quả Thực nghiệm & Định dạng Xuất bản](#8-kết-quả-thực-nghiệm--định-dạng-xuất-bản)

---

## 1. Tổng quan Bài toán & Dữ liệu

Trong vật lý năng lượng cao (High Energy Physics - HEP), việc phát hiện dấu hiệu của các hạt siêu đối xứng (SUSY) sinh ra từ máy gia tốc hạt Tevatron/LHC bị cản trở nghiêm trọng bởi các sự kiện nền (Standard Model background processes) có tín hiệu tương tự.

Tập dữ liệu **SUSY** gồm **5,000,000 sự kiện va chạm** với **18 đặc trưng liên tục** (theo Baldi et al. 2014, Nature Communications, UCI doi:10.24432/C54606):
- **8 đặc trưng động học cơ bản (Low-level features)** — đo trực tiếp từ máy dò:
  - `lepton1_pT`, `lepton1_eta`, `lepton1_phi`: Động lượng ngang, độ giả nhanh ($\eta$), góc phương vị ($\phi$) của lepton 1.
  - `lepton2_pT`, `lepton2_eta`, `lepton2_phi`: Các giá trị tương ứng của lepton 2.
  - `MET_magnitude`, `MET_phi`: Độ lớn và góc phương vị của năng lượng khuyết (Missing Transverse Energy).
- **10 đặc trưng tổ hợp bậc cao (High-level features)** — do các nhà vật lý dẫn xuất:
  - `MET_rel`, `axial_MET`, `M_R`, `M_TR_2`, `R`, `MT2`, `S_R`, `M_Delta_R`, `dPhi_r_b`, `cos_theta_r1`.

**Mục tiêu**: Dự đoán nhãn nhị phân $y \in \{0, 1\}$ ($y=1$: Sự kiện SUSY tín hiệu; $y=0$: Sự kiện nền chuẩn).

---

## 2. Cấu trúc Thư mục Dự án

```text
HGB/
├── data/
│   └── SUSY.csv                      # Tập dữ liệu 5 triệu dòng giải nén (~2.39 GB)
├── hgb_model.py                      # Thư viện thuật toán cốt lõi 100% thuần Python/NumPy
├── main.py                           # Pipeline thực thi 2-Phase, kiểm thử và xuất kết quả qua CLI
├── create_notebook.py                # Script tạo notebook.ipynb chuẩn 20 sections
├── notebook.ipynb                    # Jupyter Notebook báo cáo trực quan, biểu đồ khoa học
├── requirements.txt                  # Danh sách thư viện tối thiểu (numpy, pandas, matplotlib)
├── evaluation_summary.txt            # Báo cáo đánh giá tổng hợp
├── .gitignore                        # Cấu hình bỏ qua tệp nhị phân và dữ liệu lớn
├── outputs/                          # Thư mục lưu kết quả phân tích có cấu trúc
│   ├── metrics.json                  # Toàn bộ chỉ số, thông tin môi trường, thời gian chạy
│   ├── confusion_matrix.csv          # Ma trận nhầm lẫn chi tiết
│   ├── threshold_sweep.csv           # Bảng quét ngưỡng phân loại trên tập Validation
│   ├── feature_importance_gain.csv   # Độ quan trọng đặc trưng theo độ lợi phân tách (Gain)
│   ├── feature_importance_permutation.csv # Độ quan trọng đặc trưng theo Delta-AUC trên Validation
│   ├── training_history.csv          # Lịch sử suy giảm hàm mất mát qua từng vòng boosting
│   ├── config.json                   # Siêu tham số của mô hình
│   └── environment.json              # Thông tin phần cứng, OS, Python, NumPy, Git commit
└── tests/                            # Bộ unit test độc lập (Standard Python unittest)
    ├── test_split.py                 # Kiểm thử phân chia phân tầng & kiểm toán overlap
    ├── test_metrics.py               # Kiểm thử tính đúng đắn của toàn bộ bộ chỉ số NumPy
    ├── test_binning.py               # Kiểm thử rời rạc hóa phân vị uint8 và phát hiện NaN/Inf
    ├── test_tree.py                  # Kiểm thử xây dựng cây histogram và trọng số nút lá
    ├── test_classifier.py            # Kiểm thử bộ phân loại HGB, early stopping, full refit
    ├── test_leakage.py               # Kiểm thử ranh giới cách ly dữ liệu và chống Data Leakage
    └── test_benchmark_optimization.py # Kiểm thử tối ưu hóa vector hóa C-level & tương đương số học
```

---

## 3. Cơ sở Lý thuyết & Toán học của Thuật toán HGB

### 3.1 Khai triển Taylor bậc 2 & Bước Newton-Raphson
Với bài toán phân loại nhị phân, hàm mất mát là Binary Cross-Entropy (Log-Loss):
$$\mathcal{L}(y, F(x)) = - \Big[ y \ln(\sigma(F(x))) + (1 - y) \ln(1 - \sigma(F(x))) \Big]$$
Trong đó xác suất dự đoán $p_i = \sigma(F(x_i)) = \frac{1}{1 + e^{-\text{clip}(F(x_i), -15, 15)}}$.

Tại mỗi vòng lặp boosting $m$, thuật toán tính Gradient bậc 1 và Hessian bậc 2:
$$g_i = p_i - y_i, \quad h_i = p_i (1 - p_i)$$

Trọng số tối ưu tại nút lá $j$ có điều chuẩn $L_2$ ($\lambda$):
$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

### 3.2 Tối ưu hóa Tìm kiếm Điểm cắt $O(D \times K)$ qua Histogram
1. **Rời rạc hóa phân vị (Quantile Binning)**: Ánh xạ ma trận $X$ liên tục thành ma trận số nguyên `uint8` ($K=255$ bins).
2. **Xây dựng Histogram đơn vòng $O(N)$**:
   $$G_k = \sum_{i: x_{ij} \in \text{bin}_k} g_i, \quad H_k = \sum_{i: x_{ij} \in \text{bin}_k} h_i, \quad C_k = \sum_{i: x_{ij} \in \text{bin}_k} 1$$
3. **Tích lũy tiền tố (Prefix Sum)**: Dùng `np.cumsum` để tính thống kê nhánh trái $G_L, H_L, C_L$ và nhánh phải $G_R = G - G_L, H_R = H - H_L$:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_{\text{tot}}^2}{H_{\text{tot}} + \lambda} \right] - \gamma$$
4. Độ phức tạp tìm điểm cắt giảm từ $O(D \cdot N \log N)$ xuống **$O(D \cdot K)$**, độc lập với số mẫu dữ liệu $N$.

---

## 4. Quy trình Huấn luyện 2-Phase & Ngăn chặn Rò rỉ Dữ liệu

Dự án thiết kế cấu trúc phân bổ dữ liệu chặt chẽ nhằm triệt tiêu hoàn toàn rò rỉ thông tin (Data Leakage):

```text
TỔNG DỮ LIỆU: 5,000,000 MẪU
├── TẬP HUẤN LUYỆN & PHÁT TRIỂN (TRAIN): 4,000,000 MẪU (80%)
│   ├── PHASE 1 (DEVELOPMENT):
│   │   ├── Fit-Train Subset (90%): 3,600,000 mẫu -> Dùng để tính Gradient/Hessian & dựng cây
│   │   └── Dedicated Validation (10%): 400,000 mẫu -> Giám sát Early Stopping, tìm best_n_iter,
│   │                                                 chọn best_threshold (max F1) và đo Permutation Importance.
│   │
│   └── PHASE 2 (PRODUCTION FULL REFIT):
│       └── Huấn luyện Final Model trên TOÀN BỘ 4,000,000 mẫu Train với n_estimators = best_n_iter
│           và validation_fraction = 0.0 (Tắt Early Stopping).
│
└── TẬP KIỂM THỬ ĐỘC LẬP (TEST): 1,000,000 MẪU (20%)
    └── PHASE 3 (FINAL TEST EVALUATION):
        └── Đánh giá DUY NHẤT 1 LẦN trên 1,000,000 mẫu Test bằng Final Model với ngưỡng best_threshold đã khóa.
```

---

## 5. Chi tiết Kiến trúc Các Lớp & Hàm (Zero Scikit-Learn)

Tất cả các thành phần trong `hgb_model.py` đều được viết độc lập bằng NumPy:

1. **`train_test_split_stratified`**: Phân chia tập dữ liệu phân tầng, hỗ trợ cờ `return_indices=True` để kiểm toán rò rỉ chỉ mục.
2. **`HistBinMapper`**: Rời rạc hóa đặc trưng thành ma trận `uint8` theo phân vị đều. Chỉ `fit()` trên dữ liệu huấn luyện, kiểm tra nghiêm ngặt lỗi shape mismatch và NaN/Inf.
3. **`HistTreeNode` & `HistRegressionTree`**: Nút cây và cấu trúc cây hồi quy Histogram tối ưu hóa điểm cắt $O(D \times K)$, tôn trọng `min_samples_leaf` và `max_depth`.
4. **`CustomHistGradientBoostingClassifier`**: Bộ phân loại Ensemble hỗ trợ cả 2 chế độ:
   - `validation_fraction > 0`: Tự động tách validation phân tầng và kích hoạt Early Stopping.
   - `validation_fraction = 0.0`: Huấn luyện trên 100% dữ liệu truyền vào (Full Refit).
5. **`StratifiedKFold` & `CustomGridSearchCV`**: Tự động hóa tìm kiếm siêu tham số qua K-Fold phân tầng trên tập Train.
6. **Bộ chỉ số đánh giá (Evaluation Metrics)**:
   - `compute_confusion_matrix` (sử dụng `np.bincount` tối ưu)
   - `compute_accuracy`, `compute_precision`, `compute_recall`, `compute_specificity`, `compute_npv`, `compute_f1_score`
   - `compute_roc_auc` (theo công thức thống kê Mann-Whitney U không phụ thuộc sklearn)
   - `compute_roc_curve`, `compute_precision_recall_curve`

---

## 6. Hướng dẫn Sử dụng Dòng lệnh (CLI) & Kiểm thử

### 6.1 Chạy bộ Unit Test tự động (25 Tests)
Bộ unit test kiểm tra toàn diện tính toàn vẹn toán học, tính ổn định số, và ranh giới cách ly dữ liệu:
```bash
python -m unittest discover tests
```

### 6.2 Chạy thử nghiệm nhanh (Quick Smoke Test)
Chạy kiểm thử đường ống trên 60,000 mẫu để xác nhận toàn bộ quy trình 2-Phase hoạt động bình thường:
```bash
python main.py --nrows 60000
```

### 6.3 Chạy thử nghiệm đầy đủ trên toàn bộ 5,000,000 mẫu
```bash
python main.py --full
```

### 6.4 Chạy kết hợp tìm kiếm lưới siêu tham số (Grid Search)
```bash
python main.py --full --grid_search
```

### 6.5 Tái tạo Notebook Báo cáo 20 Sections
```bash
python create_notebook.py
```

---

## 7. Kiểm toán Rò rỉ Dữ liệu (Data Leakage Audit 14 Điểm)

Dự án tích hợp hệ thống kiểm toán tự động gồm 14 tiêu chí bắt buộc:

```text
======================== DATA LEAKAGE AUDIT ========================
[PASS] Full dataset = 5,000,000 (hoặc N mẫu thực tế)
[PASS] Train = 4,000,000 (80.0%)
[PASS] Test = 1,000,000 (20.0%)
[PASS] Unassigned = 0 (Không thất thoát mẫu)
[PASS] Coverage = 100.00%
[PASS] Train/Test index overlap = 0 (Kiểm toán giao chỉ mục độc lập)
[PASS] No NaN (Không chứa dữ liệu khuyết thiếu)
[PASS] No Inf (Không chứa giá trị vô cực)
[PASS] Bin fitting uses Train-sub only in Phase 1
[PASS] Early stopping uses Validation only in Phase 1
[PASS] Grid Search uses Training only
[PASS] Threshold selected using Validation only (Khóa ngưỡng trước khi sang Test)
[PASS] Permutation Importance uses Validation only
[PASS] Test used only for final evaluation (Tập Test chỉ mở duy nhất ở bước cuối)
[PASS] DATA LEAKAGE AUDIT PASSED
====================================================================
```

---

## 8. Kết quả Thực nghiệm & Định dạng Xuất bản

Toàn bộ kết quả thực thi được tự động lưu có cấu trúc trong thư mục `outputs/`:
- `outputs/metrics.json`: Báo cáo chỉ số toàn diện kèm metadata môi trường và Git commit.
- `outputs/confusion_matrix.csv`: Bảng ma trận nhầm lẫn ($TP, TN, FP, FN$) và tỷ lệ phần trăm.
- `outputs/threshold_sweep.csv`: Bảng đánh giá đa ngưỡng trên tập Validation.
- `outputs/feature_importance_gain.csv`: Xếp hạng đặc trưng theo độ lợi phân tách tích lũy.
- `outputs/feature_importance_permutation.csv`: Xếp hạng đặc trưng theo độ nhạy suy giảm AUC trên Validation.
- `outputs/training_history.csv`: Lịch sử hàm mất mát Log-Loss qua từng vòng lặp.
- `evaluation_summary.txt`: Báo cáo tóm tắt tổng quan dễ đọc.

---
*Tuyên bố*: Dự án này được thiết kế và triển khai theo nguyên tắc **Zero Scikit-Learn** cho toàn bộ phần thuật toán học máy lõi, đảm bảo tính nguyên bản, khả năng giải thích cao và hiệu năng tối ưu trên tập dữ liệu quy mô lớn.
