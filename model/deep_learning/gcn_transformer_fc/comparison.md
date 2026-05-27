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
| Random Forest (UOOP best FS) | **0.764 ± 0.058** | 0.904 ± 0.015 | 0.835 ± 0.025 | best FS: `All` (no FS); top feats: acidic_group, non-polar_group, X5_K. MCC 0.326 ± 0.092 |
| Naive Bayes (UOOP best FS) | 0.756 ± 0.058 | 0.818 ± 0.027 | 0.718 ± 0.036 | best FS: `Chi-square-inCV`; top feats: peptide_len, hydrophobic_janin, hydrophobic_engleman |
| LogReg (UOOP best FS) | 0.749 ± 0.055 | 0.792 ± 0.034 | 0.687 ± 0.042 | best FS: `Kruskal-inCV`; top feats: hydrophobic_janin, hydrophobic_kyte-doolittle, X5_K |
| SVM (UOOP best FS) | 0.734 ± 0.060 | 0.884 ± 0.018 | 0.805 ± 0.028 | best FS: `MW-inCV`; top feats: hydrophobic_janin, hydrophobic_eisenberg, X5_K |
| KNN (UOOP best FS) | 0.732 ± 0.060 | 0.815 ± 0.025 | 0.713 ± 0.034 | best FS: `Chi-square-inCV`; top feats: hydrophobic_janin, hydrophobic_kyte-doolittle, hydrophobic_eisenberg |
| Decision Tree (UOOP best FS) | 0.726 ± 0.051 | 0.844 ± 0.037 | 0.751 ± 0.049 | best FS: `Kruskal-inCV`; top feats: hydrophobic_janin, hydrophobic_kyte-doolittle, X5_K |
| Majority-class baseline (always predict 1) | 0.500 | 0.927 | 0.863 | Reference floor for AUC; ceiling for F1/ACC on this imbalance |

> **UOOP source:** `UOOP-Project/results/smote_in_cv/20260523_144512/feature_selection_metrics_20260523_144525.csv`.
> Each row aggregates 100 fits (10-fold × 10 repeats), so std columns are tighter than Multi_CycGT's pure 10-fold std.
> Best FS picked post-hoc by max mean AUC per classifier — mildly optimistic; Multi_CycGT's best-val-AUC pick is principled (val never updated weights), so the comparison is conservative on UOOP's side.

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

**Multimodal representation learns *something* about synthesizability, but does
not beat handcrafted-feature Random Forest.** Multi_CycGT's best-val-epoch test
AUC of **0.765 ± 0.063** is statistically indistinguishable from UOOP's best
classical baseline — Random Forest on all 233 handcrafted features at
**0.764 ± 0.058**. Every UOOP classifier with proper feature selection lands in
the 0.73–0.76 AUC band; the multimodal model joins that band rather than
exceeding it. On F1 and accuracy the picture is starker: RF reports F1 0.904 /
ACC 0.835 vs Multi_CycGT's 0.749 / 0.652 at the best-val epoch — but those
metrics reward the always-predict-positive bias an unbalanced classifier learns,
so they understate Multi_CycGT (which is more cautious on its 0.5 threshold)
more than they reflect true ranking quality. AUC is the apples-to-apples number,
and on that metric **the rich multimodal representation is not adding value over
RF on handcrafted features** for this task at N=1771.

**Why no uplift?** Three plausible explanations, in roughly decreasing
likelihood: (1) **Data scale.** 1771 samples is small for a ~2M-parameter
multimodal model; classical models with informative engineered features are
near-optimal on a dataset this size. (2) **Inductive-bias mismatch.** The graph
+ tokenized-SMILES + LogP/TPSA inputs were chosen to capture permeability
(lipophilicity, membrane interaction); synthesizability failures (aggregation,
β-sheet, difficult couplings) may correlate more with hydrophobicity *patterns*
across the sequence — which both pipelines see, but RF's `hydrophobic_janin`,
`hydrophobic_kyte-doolittle`, `acidic_group`, `peptide_len` features encode
directly. (3) **Overfitting and weak regularization.** The 500-epoch run with
no early stopping needed post-hoc epoch selection just to reach the RF
ballpark; with proper regularization (dropout, early stopping, lower LR) the
multimodal model might pull ahead, but as configured it does not.

**Practical takeaway for the writeup.** Frame Multi_CycGT as a transfer
experiment that achieved parity with the best classical baseline — not a win,
but a non-trivial result given the architecture was designed for a different
chemistry and a different endpoint. The interesting follow-ups are
(a) regularization improvements to recover the missing headroom, and
(b) feature-importance analysis of the FC branch to check whether per-residue
LogP/TPSA are actually being used (vs. the GCN/Transformer branches doing all
the work).

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
