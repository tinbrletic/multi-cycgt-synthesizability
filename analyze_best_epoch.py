"""Re-aggregate Multi_CycGT 10-fold results using best-val-AUC epoch per fold.

Rationale: training runs 500 epochs with no early stopping and overfits hard after
epoch ~100 (train_loss ~0.21, val_loss ~2.5+, test AUC peaks around epoch 100 and
degrades). Reporting final-epoch metrics catches the model deep in the overfit
regime. Selecting each fold's best-val-AUC epoch is the standard "early stopping
in post" -- principled because we never look at test labels for selection.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

PRED_DIR = Path("model/deep_learning/gcn_transformer_fc/pred_data_origin/gcn_transformer_fc")
OUT = Path("model/deep_learning/gcn_transformer_fc/results_per_fold_best.csv")


def load_preds(csv_path):
    """Return (preds, labels) for the matched-prefix rows (where predict is non-NaN)."""
    if not csv_path.exists():
        return None, None
    sub = pd.read_csv(csv_path).dropna(subset=["predict"])
    if len(sub) == 0:
        return None, None
    return sub["predict"].values, sub["true"].values.astype(int)


def safe_auc(preds, labels):
    if preds is None or labels is None or len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, preds))


rows = []
for fold in range(1, 11):
    val_aucs = np.full(500, np.nan)
    for epoch in range(1, 501):
        vp, vl = load_preds(PRED_DIR / str(fold) / "val" / f"experiment_{epoch}_predicted_valid_values.csv")
        val_aucs[epoch - 1] = safe_auc(vp, vl)

    if np.all(np.isnan(val_aucs)):
        print(f"fold {fold}: ALL val AUCs are nan -- skipping")
        continue

    best_epoch_idx = int(np.nanargmax(val_aucs))
    best_epoch = best_epoch_idx + 1
    best_val_auc = float(val_aucs[best_epoch_idx])

    # Compute test metrics at this epoch
    tp, tl = load_preds(PRED_DIR / str(fold) / "test" / f"experiment_{best_epoch}_predicted_test_values.csv")
    if tp is None:
        print(f"fold {fold}: missing test preds at epoch {best_epoch}")
        continue
    test_auc = safe_auc(tp, tl)
    test_bin = (tp >= 0.5).astype(int)
    test_f1 = float(f1_score(tl, test_bin, zero_division=0))
    test_acc = float(accuracy_score(tl, test_bin))

    # Final-epoch test AUC for comparison
    final_tp, final_tl = load_preds(PRED_DIR / str(fold) / "test" / "experiment_500_predicted_test_values.csv")
    final_test_auc = safe_auc(final_tp, final_tl)

    rows.append({
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_auc_at_best": test_auc,
        "test_f1_at_best": test_f1,
        "test_accuracy_at_best": test_acc,
        "test_auc_at_final_500": final_test_auc,
        "auc_uplift_from_early_stop": test_auc - final_test_auc,
    })

best = pd.DataFrame(rows)
best.to_csv(OUT, index=False)
print()
print("=" * 70)
print("Best-val-AUC epoch per fold")
print("=" * 70)
print(best.to_string(index=False))
print()
print("Aggregate (best-val-epoch test metrics):")
for col in ["test_auc_at_best", "test_f1_at_best", "test_accuracy_at_best"]:
    mean = best[col].mean()
    std = best[col].std()
    print(f"  {col:28s} : {mean:.4f} +/- {std:.4f}  "
          f"(min {best[col].min():.4f}, max {best[col].max():.4f})")
print()
print("Final-epoch test AUC was:", f"{best['test_auc_at_final_500'].mean():.4f} +/- {best['test_auc_at_final_500'].std():.4f}")
print("Best-epoch test AUC is :", f"{best['test_auc_at_best'].mean():.4f} +/- {best['test_auc_at_best'].std():.4f}")
print("Mean uplift            :", f"{best['auc_uplift_from_early_stop'].mean():+.4f}")
print()
print("=" * 70)
print("Pooled best-epoch test metrics (across all 10 folds)")
print("=" * 70)
all_preds = []
all_labels = []
for _, row in best.iterrows():
    tp, tl = load_preds(PRED_DIR / str(int(row.fold)) / "test" / f"experiment_{int(row.best_epoch)}_predicted_test_values.csv")
    if tp is None:
        continue
    all_preds.append(tp)
    all_labels.append(tl)
pooled_p = np.concatenate(all_preds)
pooled_l = np.concatenate(all_labels)
print(f"pooled samples : {len(pooled_p)}")
print(f"pooled AUC     : {roc_auc_score(pooled_l, pooled_p):.4f}")
pb = (pooled_p >= 0.5).astype(int)
print(f"pooled F1      : {f1_score(pooled_l, pb):.4f}")
print(f"pooled ACC     : {accuracy_score(pooled_l, pb):.4f}")
print()
print(f"Written to {OUT}")
