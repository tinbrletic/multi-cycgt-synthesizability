"""Aggregate analysis of Multi_CycGT 10-fold synthesizability training results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

RESULTS = Path("model/deep_learning/gcn_transformer_fc/results_per_fold.csv")
PRED_DIR = Path("model/deep_learning/gcn_transformer_fc/pred_data_origin/gcn_transformer_fc")

print("=" * 70)
print("PART 1: per-fold final-epoch metrics (from results_per_fold.csv)")
print("=" * 70)
df = pd.read_csv(RESULTS)
print(df.to_string(index=False))
print()
for col in ["test_auc", "test_f1", "test_accuracy", "val_auc", "val_f1", "val_accuracy"]:
    mean = df[col].mean()
    std = df[col].std()
    print(f"  {col:18s} : {mean:.4f} +/- {std:.4f}   "
          f"(min {df[col].min():.4f}, max {df[col].max():.4f})")

print()
print("=" * 70)
print("PART 2: aggregate test metrics on POOLED predictions across all 10 folds")
print("=" * 70)
print("Pooling fold-level test predictions gives a single AUC over ~1280 samples,")
print("which is more statistically meaningful than averaging per-fold AUCs.")
print()

all_preds = []
all_labels = []
for fold in range(1, 11):
    p = PRED_DIR / str(fold) / "test" / "experiment_500_predicted_test_values.csv"
    if not p.exists():
        print(f"MISSING {p}")
        continue
    sub = pd.read_csv(p).dropna(subset=["predict"])
    all_preds.append(sub["predict"].values)
    all_labels.append(sub["true"].values.astype(int))

pooled_preds = np.concatenate(all_preds)
pooled_labels = np.concatenate(all_labels)

print(f"pooled samples : {len(pooled_preds)}")
print(f"class balance  : {dict(zip(*np.unique(pooled_labels, return_counts=True)))}")
print(f"pooled AUC     : {roc_auc_score(pooled_labels, pooled_preds):.4f}")
pred_bin = (pooled_preds >= 0.5).astype(int)
print(f"pooled F1      : {f1_score(pooled_labels, pred_bin):.4f}")
print(f"pooled ACC     : {accuracy_score(pooled_labels, pred_bin):.4f}")
print(f"pooled prec    : {precision_score(pooled_labels, pred_bin, zero_division=0):.4f}")
print(f"pooled recall  : {recall_score(pooled_labels, pred_bin):.4f}")

print()
print("majority-class baseline (predict 1 always):")
maj_pred = np.ones_like(pooled_labels)
print(f"  ACC : {accuracy_score(pooled_labels, maj_pred):.4f}")
print(f"  F1  : {f1_score(pooled_labels, maj_pred):.4f}")
print(f"  AUC : 0.5000  (always-1 has no discrimination)")

print()
print("=" * 70)
print("PART 3: training-dynamics check on fold 1 (overfitting?)")
print("=" * 70)
print("Sample test-pred CSVs at epochs 1, 50, 100, 200, 300, 400, 500 for fold 1")
print()
for epoch in [1, 50, 100, 200, 300, 400, 500]:
    p = PRED_DIR / "1" / "test" / f"experiment_{epoch}_predicted_test_values.csv"
    if not p.exists():
        continue
    sub = pd.read_csv(p).dropna(subset=["predict"])
    if len(sub) == 0:
        continue
    pred_arr = sub["predict"].values
    label_arr = sub["true"].values.astype(int)
    if len(set(label_arr.tolist())) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(label_arr, pred_arr)
    bin_pred = (pred_arr >= 0.5).astype(int)
    print(f"  epoch {epoch:3d}: AUC={auc:.4f}  F1={f1_score(label_arr, bin_pred, zero_division=0):.4f}  "
          f"ACC={accuracy_score(label_arr, bin_pred):.4f}  mean_pred={pred_arr.mean():.3f}")

print()
print("=" * 70)
print("PART 4: prediction distribution at final epoch, pooled across folds")
print("=" * 70)
print("Histogram bins of sigmoid output:")
import collections
bins = np.linspace(0, 1, 11)
counts, _ = np.histogram(pooled_preds, bins=bins)
for i, c in enumerate(counts):
    bar = "#" * (c * 60 // max(counts))
    print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}) : {c:4d}  {bar}")
