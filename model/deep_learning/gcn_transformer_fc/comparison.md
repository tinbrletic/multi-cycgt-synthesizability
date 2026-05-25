# Multi_CycGT GCN-Transformer-FC vs UOOP Classical Baselines

> Multi_CycGT numbers populated from this branch's 10-fold run; UOOP baseline
> rows still need to be filled in from `feature_selection_metrics_*.csv`.

## Dataset

1771 linear peptides (Gutman et al. 2022), binary synthesis success label,
**87/13 class imbalance** (1539 positive / 232 negative).

## Evaluation protocols

| Pipeline | CV | Imbalance handling | Hyperparameters | Reported epoch |
|---|---|---|---|---|
| **Multi_CycGT (this branch)** | `KFold(n_splits=10, shuffle=True, random_state=3407)` | `WeightedRandomSampler` inside training DataLoader (no leakage) | 500 epochs, `batch_size=128`, Adam lr=1e-3, BCELoss after sigmoid | **Two reported (final-epoch + best-val-AUC); see below** |
| UOOP classical | `RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)` | SMOTE inside each train fold | Per-classifier; see UOOP-Project | n/a |

**Comparability note.** The two pipelines use different CV splits and different
random states, so per-fold scores are NOT paired. Compare only at the
mean ± std level. Multi_CycGT also evaluates on only 128 of ~177 test samples
per fold because `collate()` requires `drop_last=True` (architectural
constraint: `reshape([batch_size, 128])` in the forward pass requires a fixed
batch size). About 28% of each fold's test set is silently discarded.

## Aggregation choice — final-epoch vs best-val-AUC epoch

Multi_CycGT trains for 500 epochs with no early stopping. Per-fold trajectories
show test AUC peaking around epoch 6–22 for 8 of 10 folds, then degrading as
the model overfits (train loss falls to ~0.21 while val loss climbs to 2.5–3.7
by epoch 500 — clear 5–10× train/val loss gap = overfit signature).

Two principled choices for the headline number:

1. **Final-epoch (epoch 500).** Honest to the hyperparameter setting actually
   trained. Reports a knowably suboptimal model. AUC ≈ 0.66.
2. **Best-val-AUC epoch (early stopping in post).** For each fold, pick the
   epoch with the highest val AUC, report test AUC at that epoch. Legitimate
   because val labels were never used to update weights — equivalent to having
   run early stopping during training. AUC ≈ 0.76, +0.10 uplift.

The strong recommendation for the writeup is to **report best-val-AUC as the
headline** (with final-epoch shown alongside for transparency) — the +0.10
gap reflects an obvious overfitting problem that any reviewer would expect
us to address via early stopping.

## Headline table

| Model | Mean AUC ± std | Mean F1 ± std | Mean ACC ± std | Notes |
|---|---|---|---|---|
| **Multi_CycGT (gcn_transformer_fc), best-val-AUC epoch** | **0.765 ± 0.063** | 0.749 ± 0.113 | 0.652 ± 0.127 | Pooled (1280 samples): AUC 0.744, F1 0.759, ACC 0.652. Per-fold best epochs: 6, 7, 10, 300, 22, 15, 14, 94, 15, 2 (median 14.5) |
| Multi_CycGT (gcn_transformer_fc), final epoch 500 | 0.661 ± 0.089 | 0.868 ± 0.032 | 0.781 ± 0.046 | Pooled (1280 samples): AUC 0.668, F1 0.869, ACC 0.781. Reported for transparency; model is overfit by this epoch |
| LogReg (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| Random Forest (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| SVM (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| KNN (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| Decision Tree (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| Naive Bayes (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| Majority-class baseline (always predict 1) | 0.500 | 0.927 | 0.863 | Reference floor for AUC; ceiling for F1/ACC on this imbalance |

## Per-fold breakdown (best-val-AUC epoch)

| Fold | Best epoch | Val AUC | Test AUC | Test F1 | Test ACC |
|---|---|---|---|---|---|
| 1 | 6 | 0.802 | 0.833 | 0.733 | 0.625 |
| 2 | 7 | 0.816 | 0.751 | 0.741 | 0.617 |
| 3 | 10 | 0.853 | 0.853 | 0.757 | 0.648 |
| 4 | 300 | 0.771 | 0.682 | 0.829 | 0.742 |
| 5 | 22 | 0.829 | 0.723 | 0.605 | 0.500 |
| 6 | 15 | 0.775 | 0.825 | 0.896 | 0.828 |
| 7 | 14 | 0.808 | 0.674 | 0.595 | 0.469 |
| 8 | 94 | 0.877 | 0.737 | 0.874 | 0.797 |
| 9 | 15 | 0.867 | 0.766 | 0.844 | 0.758 |
| 10 | 2 | 0.810 | 0.804 | 0.615 | 0.531 |
| **mean ± std** | — | 0.821 ± 0.038 | **0.765 ± 0.063** | 0.749 ± 0.113 | 0.652 ± 0.127 |

## Observations

**Multimodal representation learns *something* about synthesizability.**
Best-val-epoch test AUC of 0.76 is well above the 0.50 chance line. The model
ranks a randomly chosen successful peptide above a randomly chosen failed one
about 76% of the time. Discrimination is real but not strong — classical models
with rich hand-crafted features (440 sequence-composition features +
physicochemical descriptors + SMOTE) may match or beat this depending on the
choice of classifier and feature-selection strategy. Whether the multimodal
graph + sequence + descriptors representation captures complementary
information to UOOP's handcrafted features is the central question this table
will answer — fill in the UOOP rows to settle it.

**Overfitting is the dominant failure mode at N=1771.** Training for 500
epochs with no regularization is too aggressive: 8 of 10 folds peak by epoch
22. Per-fold val/test AUC trajectories show clear bias-variance crossover early
in training. The model's three-branch fusion has ~2M+ parameters and the
fused head `fc1 = Linear(16552, 1)` is a particularly unregularized bottleneck.
Likely fixes (in order of cost): proper early stopping during training; lower
LR (1e-4) with 100 epochs; dropout on the fused head; replacement of
WeightedRandomSampler with focal loss for a stronger minority-class gradient
signal without resampling artifacts.

**Calibration is poor — predictions are bimodal and overconfident.** At the
final epoch, 68% of pooled test predictions land in [0.9, 1.0) and another 10%
in [0.0, 0.1); only 1% land in the [0.4, 0.5) "uncertain" bin. Best-val-epoch
predictions are better spread but still polarized. If you intend to use the
model for ranking (e.g. prioritizing peptides for wet-lab attempt), AUC is the
right metric; if you intend to use raw probabilities (e.g. calibrated risk
scores for triage), the model needs temperature scaling or Platt calibration
on a held-out set.

**Cross-task transfer caveat (genuine, not a hedge).** Multi_CycGT was
designed for cyclic peptides + membrane permeability. We're applying the same
architecture to linear peptides + synthesis success — different chemistry,
different failure modes (aggregation, β-sheet propensity, difficult couplings).
A multimodal architecture that captures lipophilicity + TPSA + H-bond patterns
well may not capture synthesis-failure-relevant features. The achieved AUC ~0.76
is consistent with this: the architecture is informative but the inductive
biases are not perfectly aligned with the task.

## Hardware and runtime

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (Ampere, compute capability 8.6)
- torch 2.0.1+cu117, dgl 1.1.0+cu117, dgllife 0.3.2 (patched), Python 3.10 venv
- Total training time for 10 folds × 500 epochs: ~3–6 hours
  (fill in actual from training_log.txt timestamps if available)

## Reproducing the numbers

Both aggregation files are produced from the per-epoch prediction CSVs in
`pred_data_origin/`:

```powershell
# Final-epoch (already done during training, written by model_concat.py)
Get-Content model/deep_learning/gcn_transformer_fc/results_per_fold.csv

# Best-val-AUC epoch (post-hoc re-aggregation)
python analyze_best_epoch.py
# Writes model/deep_learning/gcn_transformer_fc/results_per_fold_best.csv
```

`analyze_results.py` and `analyze_best_epoch.py` live at the repo root.
