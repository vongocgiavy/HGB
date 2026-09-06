"""
Pipeline thực thi và đánh giá mô hình Histogram Gradient Boosting (HGB)
trên tập dữ liệu va chạm hạt SUSY (UCI Benchmark).
Triển khai 100% bằng Python thuần và NumPy -- Không phụ thuộc Scikit-Learn.
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from hgb_model import (
    CustomHistGradientBoostingClassifier,
    train_test_split_stratified,
    compute_confusion_matrix,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_specificity,
    compute_npv,
    compute_f1_score,
    compute_roc_auc,
    StratifiedKFold,
    cross_val_score,
    CustomGridSearchCV,
)

# ==============================================================================
# FEATURE DEFINITIONS  (SUSY dataset, Baldi et al. 2014, UCI doi:10.24432/C54606)
# Col 0 = label (1=SUSY signal, 0=SM background)
# Col 1-8  = 8 low-level kinematic features (measured by detector)
# Col 9-18 = 10 high-level derived features (Razor / Super-Razor variables)
# ==============================================================================
FEATURE_NAMES = [
    "lepton1_pT",   "lepton1_eta",  "lepton1_phi",
    "lepton2_pT",   "lepton2_eta",  "lepton2_phi",
    "MET_magnitude","MET_phi",
    "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2", "S_R",
    "M_Delta_R", "dPhi_r_b", "cos_theta_r1",
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

SEP  = "=" * 78
LINE = "-" * 78


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Huan luyen HGB cho bai toan SUSY (Zero Scikit-Learn)."
    )
    p.add_argument("--nrows",            type=int,   default=60_000)
    p.add_argument("--n_estimators",     type=int,   default=200)
    p.add_argument("--lr",               type=float, default=0.08)
    p.add_argument("--max_depth",        type=int,   default=6)
    p.add_argument("--min_samples_leaf", type=int,   default=30)
    p.add_argument("--l2_reg",           type=float, default=1.0)
    p.add_argument("--max_bins",         type=int,   default=255)
    p.add_argument("--min_gain",         type=float, default=0.0)
    p.add_argument("--val_fraction",     type=float, default=0.1)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--tol",              type=float, default=1e-4)
    p.add_argument("--threshold",        type=float, default=0.40)
    p.add_argument("--random_state",     type=int,   default=42)
    p.add_argument("--grid_search",      action="store_true")
    p.add_argument("--cv_folds",         type=int,   default=3)
    return p.parse_args()


def main():
    args = parse_arguments()

    print()
    print(SEP)
    print("   SUSY PARTICLE COLLISION CLASSIFICATION")
    print("   Algorithm : Histogram Gradient Boosting (100% NumPy, Zero Sklearn)")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_dir  = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, "data", "SUSY.csv")
    if not os.path.exists(data_path):
        alt_path = os.path.join(data_dir, "SUSY.csv")
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            sys.exit(f"[!] File not found: {data_path} or {alt_path}")

    nrows = None if (args.nrows is None or args.nrows <= 0) else args.nrows
    nrows_str = "ALL 5,000,000" if nrows is None else f"{nrows:,}"
    print(f"\n[1] Loading {nrows_str} rows from {data_path} ...")
    t_load = time.time()
    df = pd.read_csv(data_path, header=None, nrows=nrows)
    df.columns = ["label"] + FEATURE_NAMES
    t_load = time.time() - t_load

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"    Read time : {t_load:.2f}s")
    print(f"    Shape     : {X.shape[0]:,} x {X.shape[1]}")
    print(f"    Labels    : {n_pos:,} SUSY signal ({n_pos/len(y)*100:.2f}%)"
          f"  |  {n_neg:,} background ({n_neg/len(y)*100:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Train / Test split (stratified)
    # ------------------------------------------------------------------
    print(f"\n[2] Stratified Train/Test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=0.2, random_state=args.random_state
    )
    print(f"    Train : {X_train.shape[0]:,}  (pos rate {np.mean(y_train)*100:.2f}%)")
    print(f"    Test  : {X_test.shape[0]:,}  (pos rate {np.mean(y_test)*100:.2f}%)")

    # ------------------------------------------------------------------
    # 3. Hyperparameters
    # ------------------------------------------------------------------
    hgb_config = {
        "n_estimators":        args.n_estimators,
        "learning_rate":       args.lr,
        "max_depth":           args.max_depth,
        "min_samples_leaf":    args.min_samples_leaf,
        "l2_regularization":   args.l2_reg,
        "max_bins":            args.max_bins,
        "min_gain_to_split":   args.min_gain,
        "validation_fraction": args.val_fraction,
        "n_iter_no_change":    args.patience,
        "tol":                 args.tol,
        "random_state":        args.random_state,
    }

    print("\n[3] Hyperparameters:")
    print("    " + "-" * 50)
    for k, v in hgb_config.items():
        print(f"    {k:<22} = {v}")
    print("    " + "-" * 50)

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    if args.grid_search:
        print("\n" + SEP)
        print("   GRID SEARCH K-FOLD CV (CUSTOM, ZERO SKLEARN)")
        print(SEP)
        param_grid = {
            "learning_rate":    [0.05, 0.1],
            "max_depth":        [4, 6],
            "min_samples_leaf": [20, 50],
        }
        base = CustomHistGradientBoostingClassifier(
            n_estimators=min(args.n_estimators, 80),
            l2_regularization=args.l2_reg,
            max_bins=args.max_bins,
            validation_fraction=args.val_fraction,
            n_iter_no_change=args.patience,
            random_state=args.random_state,
        )
        gs = CustomGridSearchCV(
            estimator=base, param_grid=param_grid,
            cv=args.cv_folds, scoring="roc_auc",
            threshold=args.threshold, refit=True, verbose=1,
        )
        t0 = time.time()
        gs.fit(X_train, y_train)
        t_train = time.time() - t0
        print("\nGrid Search Results (sorted by ROC-AUC):")
        print(gs.summary())
        print(f"\nBest params : {gs.best_params_}")
        model = gs.best_estimator_
    else:
        print("\n[4] Training HGB (with internal Early Stopping)...")
        sys.stdout.flush()
        model = CustomHistGradientBoostingClassifier(**hgb_config)
        t0 = time.time()
        model.fit(X_train, y_train, verbose=True)
        t_train = time.time() - t0
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # 5. Predict & Metrics
    # ------------------------------------------------------------------
    print(f"\n[5] Predicting on Test set (threshold = {args.threshold:.2f}) ...")
    y_proba = model.predict_proba(X_test)
    y_pred  = model.predict(X_test, threshold=args.threshold)

    acc  = compute_accuracy(y_test, y_pred)
    prec = compute_precision(y_test, y_pred)
    rec  = compute_recall(y_test, y_pred)
    spec = compute_specificity(y_test, y_pred)
    npv  = compute_npv(y_test, y_pred)
    f1   = compute_f1_score(y_test, y_pred)
    auc  = compute_roc_auc(y_test, y_proba)
    tp, tn, fp, fn = compute_confusion_matrix(y_test, y_pred)

    total = tp + tn + fp + fn
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print()
    print(SEP)
    print(f"   EVALUATION RESULTS  (threshold = {args.threshold:.2f})")
    print(SEP)
    metrics = [
        ("Accuracy",    acc,  f"{acc*100:.2f}%  all events classified correctly"),
        ("Precision",   prec, f"{prec*100:.2f}%  of predicted SUSY are true SUSY"),
        ("Recall",      rec,  f"{rec*100:.2f}%  of true SUSY events detected"),
        ("Specificity", spec, f"{spec*100:.2f}%  background correctly rejected"),
        ("NPV",         npv,  f"{npv*100:.2f}%  of predicted background are true bkg"),
        ("F1-Score",    f1,   "Harmonic mean of Precision & Recall"),
        ("ROC-AUC",     auc,  "Probability-level separation power"),
    ]
    for name, val, desc in metrics:
        print(f"  {name:<14} {val:8.4f}   {desc}")
    print("  " + LINE)

    print("\n  CONFUSION MATRIX:")
    print(f"  {'':26}| {'Pred: Background':^18} | {'Pred: SUSY':^18}")
    print(f"  {LINE}")
    print(f"  {'True: Background':<26}| TN = {tn:6,} ({tn/total*100:5.1f}%)  | FP = {fp:6,} ({fp/total*100:5.1f}%)")
    print(f"  {'True: SUSY':<26}| FN = {fn:6,} ({fn/total*100:5.1f}%)  | TP = {tp:6,} ({tp/total*100:5.1f}%)")
    print(f"  False Positive Rate : {fpr*100:.2f}%   |   False Negative Rate : {fnr*100:.2f}%")

    # ------------------------------------------------------------------
    # 7. Threshold sweep
    # ------------------------------------------------------------------
    print()
    print(SEP)
    print("   THRESHOLD SWEEP  (adjust Recall without retraining)")
    print(SEP)
    print(f"  {'Thr':>5} | {'Acc':>8} | {'Prec':>8} | {'Recall':>8} |"
          f" {'F1':>8} | {'Spec':>8} | {'FN':>6}")
    print("  " + LINE)
    for th in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        p_th = (y_proba >= th).astype(int)
        _, _, _, fn_th = compute_confusion_matrix(y_test, p_th)
        marker = " <==CURRENT" if abs(th - args.threshold) < 1e-4 else ""
        print(f"  {th:.2f}  | {compute_accuracy(y_test,p_th)*100:7.2f}%"
              f" | {compute_precision(y_test,p_th)*100:7.2f}%"
              f" | {compute_recall(y_test,p_th)*100:7.2f}%"
              f" | {compute_f1_score(y_test,p_th)*100:7.2f}%"
              f" | {compute_specificity(y_test,p_th)*100:7.2f}%"
              f" | {fn_th:5,}{marker}")
    print("  " + LINE)
    print("  Tip: Lower threshold => higher Recall (fewer missed SUSY).")
    print("       --threshold 0.30  for Recall >80%  |  --threshold 0.50 for Precision >80%")

    # ------------------------------------------------------------------
    # 8. Feature Importances (Gain + Permutation)
    # ------------------------------------------------------------------
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    rng      = np.random.default_rng(args.random_state)
    perm_imp = np.zeros(len(FEATURE_NAMES))
    for j in range(len(FEATURE_NAMES)):
        Xp = X_test.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        perm_imp[j] = auc - compute_roc_auc(y_test, model.predict_proba(Xp))
    perm_imp    = np.clip(perm_imp, 0.0, None)
    perm_sorted = np.argsort(perm_imp)[::-1]

    print()
    print(SEP)
    print("   FEATURE IMPORTANCES  (Gain = node-split accumulation | Perm = Delta AUC)")
    print(SEP)
    print(f"  {'#':>3} | {'Feature':<16} | {'Gain%':>7} | {'Perm#':>5} | {'DeltaAUC':>9} | Description")
    print("  " + LINE)
    for rank, idx in enumerate(sorted_idx, 1):
        feat   = FEATURE_NAMES[idx]
        g      = importances[idx]
        p      = perm_imp[idx]
        p_rank = int(np.where(perm_sorted == idx)[0][0]) + 1
        desc   = FEATURE_DESCRIPTIONS.get(feat, "")
        print(f"  {rank:3d} | {feat:<16} | {g*100:6.2f}% | #{p_rank:<4} | {p:+.5f}  | {desc}")
    print("  " + LINE)
    print("  NOTE: Gain can be inflated for features chosen at root nodes.")
    print("        Permutation DeltaAUC is model-agnostic and more reliable.")

    # ------------------------------------------------------------------
    # 9. Training diagnostics
    # ------------------------------------------------------------------
    stopped = getattr(model, "stopped_iter_", model.n_iter_)
    print()
    print(SEP)
    print("   TRAINING DIAGNOSTICS")
    print(SEP)
    print(f"  Max estimators configured : {args.n_estimators}")
    print(f"  Trees built (stopped at)  : {stopped}")
    print(f"  Best iteration (pruned to): {model.best_n_iter_}")
    print(f"  Best val loss             : {model.best_val_loss_:.5f}")
    print(f"  Training time             : {t_train:.2f}s  ({t_train/stopped:.3f}s/tree)")
    print(SEP)

    # ------------------------------------------------------------------
    # 10. Save report
    # ------------------------------------------------------------------
    report_path = os.path.join(data_dir, "evaluation_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  HGB EVALUATION REPORT -- SUSY DATASET\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data    : SUSY.csv  {nrows_str} rows  "
                f"Train={X_train.shape[0]:,}  Test={X_test.shape[0]:,}\n")
        f.write(f"Model   : Histogram Gradient Boosting (Zero Scikit-Learn)\n")
        f.write(f"Seed    : {args.random_state}\n\n")
        f.write("[HYPERPARAMETERS]:\n")
        for k, v in hgb_config.items():
            f.write(f"  {k:<22}: {v}\n")
        f.write("\n[TEST SET METRICS]:\n")
        for name, val, _ in metrics:
            f.write(f"  {name:<14}: {val:.4f}\n")
        f.write(f"  Confusion  : TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")
        f.write(f"  Best round : {model.best_n_iter_}/{stopped}"
                f"  (ValLoss={model.best_val_loss_:.5f})\n\n")
        f.write("[TOP 5 BY GAIN]:\n")
        for r, i in enumerate(sorted_idx[:5], 1):
            f.write(f"  {r}. {FEATURE_NAMES[i]:<16} Gain={importances[i]*100:.2f}%"
                    f"  DeltaAUC={perm_imp[i]:+.5f}\n")
        f.write("\n[TOP 5 BY PERMUTATION (DELTA AUC)]:\n")
        for r, i in enumerate(perm_sorted[:5], 1):
            f.write(f"  {r}. {FEATURE_NAMES[i]:<16} DeltaAUC={perm_imp[i]:+.5f}"
                    f"  Gain={importances[i]*100:.2f}%\n")

    print(f"\n[v] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
