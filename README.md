# Phân loại Hạt Siêu Đối Xứng (SUSY) bằng Histogram Gradient Boosting (HGB)

Dự án nghiên cứu và triển khai hoàn chỉnh thuật toán **Histogram Gradient Boosting (HGB)** từ đầu 100% bằng **Python thuần và NumPy** (Zero Scikit-Learn trong toàn bộ thuật toán cốt lõi và hệ thống metrics). Mô hình được tối ưu hóa chuyên sâu để phân loại các sự kiện va chạm hạt siêu đối xứng (Supersymmetric Particles) trên tập dữ liệu chuẩn quốc tế **SUSY Benchmark** từ **UCI Machine Learning Repository** (5,000,000 mẫu dữ liệu).

---

## Mục lục
1. [Tổng quan Bài toán & Dữ liệu](#1-tổng-quan-bài-toán--dữ-liệu)
2. [Cấu trúc Thư mục Dự án](#2-cấu-trúc-thư-mục-dự-án)
3. [Quy trình 21 Bước Xây dựng Mô hình Machine Learning](#3-quy-trình-21-bước-xây-dựng-mô-hình-machine-learning)
4. [Cơ sở Lý thuyết & Toán học của Thuật toán HGB](#4-cơ-sở-lý-thuyết--toán-học-của-thuật-toán-hgb)
5. [Quy trình Huấn luyện 2-Phase & Ngăn chặn Rò rỉ Dữ liệu](#5-quy-trình-huấn-luyện-2-phase--ngăn-chặn-rò-rỉ-dữ-liệu)
6. [Chi tiết Kiến trúc Các Lớp & Hàm (Zero Scikit-Learn)](#6-chi-tiết-kiến-trúc-các-lớp--hàm-zero-scikit-learn)
7. [Hướng dẫn Sử dụng Dòng lệnh (CLI) & Kiểm thử](#7-hướng-dẫn-sử-dụng-dòng-lệnh-cli--kiểm-thử)
8. [Kiểm toán Rò rỉ Dữ liệu (Data Leakage Audit 14 Điểm)](#8-kiểm-toán-rò-rỉ-dữ-liệu-data-leakage-audit-14-điểm)
9. [Kết quả Thực nghiệm & Định dạng Xuất bản](#9-kết-quả-thực-nghiệm--định-dạng-xuất-bản)

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
│   └── SUSY.csv                      # Tập dữ liệu 5 triệu dòng (~2.39 GB) hoặc mẫu mô phỏng
├── hgb_model.py                      # Thư viện thuật toán cốt lõi 100% thuần Python/NumPy
├── main.py                           # Pipeline thực thi 2-Phase, kiểm thử và xuất kết quả qua CLI
├── build_21_steps_notebook.py        # Script tạo notebook.ipynb chuẩn 21 bước Machine Learning
├── notebook.ipynb                    # Jupyter Notebook triển khai trọn vẹn 21 bước Machine Learning
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
```

---

## 3. Quy trình 21 Bước Xây dựng Mô hình Machine Learning

Dự án và tệp `notebook.ipynb` được thiết kế cấu trúc chặt chẽ theo **21 bước chuẩn mực**:

```text
Problem Definition ──> Domain Understanding ──> Data Cleaning ──> Feature Processing ──> Feature Engineering
         │
         ▼
   Data Splitting ──> Baseline Model ──> Model Selection ──> Training (2-Phase) ──> Hyperparameter Tuning
         │
         ▼
  Cross-Validation ──> Evaluation Metrics ──> Statistical Validation ──> Error Analysis
         │
         ▼
  Model Interpretability ──> Iterative Improvement Cycle (Loop Back & Refine)
```

| STT | Tên bước chuẩn hóa | Nội dung triển khai trong dự án |
| :---: | :--- | :--- |
| **1** | **Xác định bài toán (Problem Definition)** | Phân loại sự kiện va chạm hạt SUSY tín hiệu ($y=1$) vs Nền chuẩn SM ($y=0$), 18 biến đầu vào. |
| **2** | **Xác định bản chất bài toán ML** | Supervised Binary Classification trên Tabular Data, Offline Batch Training, Sub-millisecond Inference. |
| **3** | **Khảo sát lĩnh vực & không gian dữ liệu** | Phân tích tri thức HEP: 8 biến động học cơ bản (Lepton $p_T, \eta, \phi$, $\text{MET}$) và 10 biến Razor bất biến khối lượng. |
| **4** | **Khám phá và xử lý dữ liệu (EDA & Cleaning)** | Kiểm toán 0 NaN, 0 Inf, kiểm tra trùng lặp và xác nhận tỷ lệ lớp cân bằng tự nhiên (~45.8% vs 54.2%). |
| **5** | **Chuẩn hóa đặc trưng (Feature Scaling)** | Phân tích tính chất Scale-invariant của cây; ứng dụng Quantile Binning (`HistBinMapper` 255 bins uint8). |
| **6** | **Xử lý biến phân loại (Categorical Data)** | Kiểm tra kiểu dữ liệu: 100% đặc trưng là số thực liên tục (`float32`), không cần One-Hot Encoding. |
| **7** | **Lựa chọn thuật toán & hàm mất mát** | Histogram Gradient Boosting tối ưu $O(D \cdot K)$, hàm mất mát Log-Loss, Gradient/Hessian bậc 2 và $L_2$ regularization. |
| **8** | **Kỹ thuật tạo đặc trưng (Feature Engineering)** | Khảo sát tương tác vật lý: tỷ số $p_T$, chênh lệch góc mở $\Delta \phi$, tổng năng lượng vô hướng $H_T$. |
| **9** | **Chia dữ liệu & kiểm soát Data Leakage** | Kiểm toán giao thoa chỉ mục (0 Overlap), độc lập tuyệt đối giữa Train, Validation và Test. |
| **10** | **Lựa chọn phương pháp chia dữ liệu** | Áp dụng Stratified Split bảo toàn phân phối nhãn I.I.D trên cả Train (80%) và Test (20%). |
| **11** | **Xây dựng mô hình cơ sở (Baseline Model)** | Thiết lập Dummy Majority Baseline (ROC-AUC=0.5) và Logistic Regression Baseline thuần NumPy (~0.78 ROC-AUC). |
| **12** | **Nguyên lý No Free Lunch** | Biện minh khoa học: HGB vượt trội Linear về phi tuyến và vượt trội DNN về tốc độ/tài nguyên trên dữ liệu bảng. |
| **13** | **Phân tích Bias và Variance** | Khảo sát đánh đổi Underfitting vs Overfitting; kiểm soát qua độ sâu cây (`max_depth=6`), $L_2$ penalty và Early Stopping. |
| **14** | **Lựa chọn & đánh giá Evaluation Metrics** | Đo lường toàn diện ROC-AUC, PR-AUC, F1; thuật toán quét ngưỡng (Threshold Tuning) trên Validation để khóa $T^*$. |
| **15** | **Kiểm định chéo K-fold (Cross-Validation)** | `StratifiedKFold` ($K=3$) và `cross_val_score` thuần NumPy đánh giá độ ổn định $\mu \pm \sigma$. |
| **16** | **Tối ưu siêu tham số (Tuning)** | `CustomGridSearchCV` thuần NumPy tìm kiếm bộ tham số tối ưu và ứng dụng Early Stopping tiết kiệm tính toán. |
| **17** | **Thực nghiệm huấn luyện mô hình** | Bảng nhật ký thực nghiệm đa cấu hình; Quy trình 2-Phase (Phase 1 Dev tìm `best_n_iter` -> Phase 2 Full Refit -> Phase 3 Test). |
| **18** | **Kiểm định thống kê độ tin cậy** | Bootstrapping 1,000 lần ước lượng Khoảng tin cậy 95% CI; chứng minh HGB vượt trội có ý nghĩa so với Baseline ($p < 0.001$). |
| **19** | **Phân tích lỗi (Error Analysis)** | Bóc tách ma trận nhầm lẫn: phân tích False Positives, False Negatives và mẫu ranh giới không chắc chắn ($p \approx 0.5$). |
| **20** | **Khả năng giải thích mô hình (Interpretability)** | Trực quan hóa Split Gain Importance và Permutation Importance; xác nhận biến `MET_magnitude` dẫn đầu độ quan trọng. |
| **21** | **Chu trình lặp cải tiến mô hình** | Thiết lập vòng lặp phản hồi cải tiến liên tục: Đánh giá $\to$ Phân tích lỗi $\to$ Feature $\to$ Tối ưu $\to$ Triển khai. |

---

## 4. Cơ sở Lý thuyết & Toán học của Thuật toán HGB

### 4.1 Khai triển Taylor bậc 2 & Bước Newton-Raphson
Với bài toán phân loại nhị phân, hàm mất mát là Binary Cross-Entropy (Log-Loss):
$$\mathcal{L}(y, F(x)) = - \Big[ y \ln(\sigma(F(x))) + (1 - y) \ln(1 - \sigma(F(x))) \Big]$$
Trong đó xác suất dự đoán $p_i = \sigma(F(x_i)) = \frac{1}{1 + e^{-\text{clip}(F(x_i), -15, 15)}}$.

Tại mỗi vòng lặp boosting $m$, thuật toán tính Gradient bậc 1 và Hessian bậc 2:
$$g_i = p_i - y_i, \quad h_i = p_i (1 - p_i)$$

Trọng số tối ưu tại nút lá $j$ có điều chuẩn $L_2$ ($\lambda$):
$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

### 4.2 Tối ưu hóa Tìm kiếm Điểm cắt $O(D \times K)$ qua Histogram
1. **Rời rạc hóa phân vị (Quantile Binning)**: Ánh xạ ma trận $X$ liên tục thành ma trận số nguyên `uint8` ($K=255$ bins).
2. **Xây dựng Histogram đơn vòng $O(N)$**:
   $$G_k = \sum_{i: x_{ij} \in \text{bin}_k} g_i, \quad H_k = \sum_{i: x_{ij} \in \text{bin}_k} h_i, \quad C_k = \sum_{i: x_{ij} \in \text{bin}_k} 1$$
3. **Tích lũy tiền tố (Prefix Sum)**: Dùng `np.cumsum` để tính thống kê nhánh trái $G_L, H_L, C_L$ và nhánh phải $G_R = G - G_L, H_R = H - H_L$:
   $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_{\text{tot}}^2}{H_{\text{tot}} + \lambda} \right] - \gamma$$
4. Độ phức tạp tìm điểm cắt giảm từ $O(D \cdot N \log N)$ xuống **$O(D \cdot K)$**, độc lập với số mẫu dữ liệu $N$.

---

## 5. Quy trình Huấn luyện 2-Phase & Ngăn chặn Rò rỉ Dữ liệu

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

## 6. Chi tiết Kiến trúc Các Lớp & Hàm (Zero Scikit-Learn)

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

## 7. Hướng dẫn Sử dụng Dòng lệnh (CLI) & Kiểm thử

### 7.1 Chạy bộ Unit Test tự động (25 Tests)
Bộ unit test kiểm tra toàn diện tính toàn vẹn toán học, tính ổn định số, và ranh giới cách ly dữ liệu:
```bash
python -m unittest discover tests
```

### 7.2 Chạy thử nghiệm nhanh (Quick Smoke Test)
Chạy kiểm thử đường ống trên 60,000 mẫu để xác nhận toàn bộ quy trình 2-Phase hoạt động bình thường:
```bash
python main.py --nrows 60000
```

### 7.3 Chạy thử nghiệm đầy đủ trên toàn bộ 5,000,000 mẫu
```bash
python main.py --full
```

### 7.4 Chạy kết hợp tìm kiếm lưới siêu tham số (Grid Search)
```bash
python main.py --full --grid_search
```

### 7.5 Tái tạo Notebook Báo cáo Chuẩn 21 Bước Machine Learning
```bash
python build_21_steps_notebook.py
```

---

## 8. Kiểm toán Rò rỉ Dữ liệu (Data Leakage Audit 14 Điểm)

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

## 9. Kết quả Thực nghiệm & Định dạng Xuất bản

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
