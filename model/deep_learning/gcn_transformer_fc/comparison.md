# Multi_CycGT GCN-Transformer-FC vs UOOP Classical Baselines

> Multi_CycGT numbers are from this branch's 10-fold run **with early stopping
> enabled** (val loss, patience 50, keep-last epoch, 500-epoch safety ceiling).
> UOOP baseline rows are from `feature_selection_metrics_*.csv`.

## Dataset

1771 linear peptides (Gutman et al. 2022), binary synthesis success label,
**87/13 class imbalance** (1539 positive / 232 negative).

## Evaluation protocols

| Pipeline | CV | Imbalance handling | Hyperparameters | Reported epoch |
|---|---|---|---|---|
| **Multi_CycGT (this branch)** | `KFold(n_splits=10, shuffle=True, random_state=3407)` | `WeightedRandomSampler` inside training DataLoader (no leakage) | Adam lr=1e-3, `batch_size=128`, BCELoss after sigmoid; **early stopping on val loss, patience 50, 500-epoch ceiling** | **Stop epoch (the kept model); folds stopped at epoch 52–68** |
| UOOP classical | `RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)` | SMOTE inside each train fold | Per-classifier; see UOOP-Project | n/a |

**Comparability note.** The two pipelines use different CV splits and different
random states, so per-fold scores are NOT paired. Compare only at the
mean ± std level. Multi_CycGT also evaluates on only 128 of ~177 test samples
per fold because `collate()` requires `drop_last=True` (architectural
constraint: `reshape([batch_size, 128])` in the forward pass requires a fixed
batch size). About 28% of each fold's test set is silently discarded; the
pooled metrics below therefore cover 1280 samples (128 × 10), not the full ~1771.

## Early stopping — what gets reported now

Earlier runs trained a fixed 500 epochs with no early stopping and overfit hard
(train loss fell to ~0.21 while val loss climbed to 2.5–3.7; final-epoch test
AUC collapsed to ~0.66). Reporting that run required **post-hoc** selection of
each fold's best-validation epoch just to reach AUC ~0.76 — defensible (val
labels never update weights) but awkward, because the reported model wasn't the
model the training loop produced.

This run adds **early stopping during training** (`early_stopping.EarlyStopping`,
mode=`min` on validation loss, patience 50, keep-last weights). Every fold
stopped well before the 500 ceiling, at exactly `best_val_loss_epoch + 50`:

| Fold | Best-val-loss epoch | Stop epoch |
|---|---|---|
| 1 | 7 | 57 |
| 2 | 3 | 53 |
| 3 | 18 | 68 |
| 4 | 5 | 55 |
| 5 | 6 | 56 |
| 6 | 2 | 52 |
| 7 | 8 | 58 |
| 8 | 10 | 60 |
| 9 | 17 | 67 |
| 10 | 3 | 53 |

**Result: early stopping recovers the headroom automatically.** The kept
(stop-epoch) model scores test AUC **0.757 ± 0.052**, matching the old post-hoc
best-val number (0.765) and far above the old overfit final-epoch (0.661).
Re-running best-val-epoch selection *on top of* this run adds only **+0.001 AUC**
(0.758 vs 0.757) and actually *hurts* F1/accuracy (best-val-AUC epochs have a
poorly-calibrated 0.5 threshold). So the stop-epoch model is reported as the
headline — it is the model the loop actually produced, and post-hoc selection no
longer buys anything.

> Note: val loss bottoms very early (epoch 2–18), so keep-last keeps a model ~50
> epochs past the val-loss minimum. Test *AUC* (a ranking metric) is robust to
> this; val *loss* keeps rising because the model grows overconfident
> (a calibration effect, not a ranking one — see "Calibration" below).

## Headline table

| Model | Mean AUC ± std | Mean F1 ± std | Mean ACC ± std | Notes |
|---|---|---|---|---|
| **Multi_CycGT (gcn_transformer_fc), early stopping** | **0.757 ± 0.052** | 0.814 ± 0.072 | 0.720 ± 0.083 | Pooled (1280 samples): AUC 0.751, F1 0.819, ACC 0.720, prec 0.925, recall 0.735. Stop epochs 52–68 (median 56.5) |
| Multi_CycGT, post-hoc best-val-AUC epoch (same run) | 0.758 ± 0.072 | 0.713 ± 0.186 | 0.631 ± 0.187 | Pooled (1280): AUC 0.726, F1 0.744, ACC 0.631. Shown only to demonstrate early stopping already captures the optimum (+0.001 AUC); F1/ACC are worse and erratic due to threshold miscalibration at val-AUC-best epochs |
| Random Forest (UOOP best FS) | **0.764 ± 0.058** | 0.904 ± 0.015 | 0.835 ± 0.025 | best FS: `All` (no FS); top feats: acidic_group, non-polar_group, X5_K. MCC 0.326 ± 0.092 |
| Naive Bayes (UOOP best FS) | 0.756 ± 0.058 | 0.818 ± 0.027 | 0.718 ± 0.036 | best FS: `Chi-square-inCV`; top feats: peptide_len, hydrophobic_janin, hydrophobic_engleman |
| LogReg (UOOP best FS) | 0.749 ± 0.055 | 0.792 ± 0.034 | 0.687 ± 0.042 | best FS: `Kruskal-inCV`; top feats: hydrophobic_janin, hydrophobic_kyte-doolittle, X5_K |
| SVM (UOOP best FS) | 0.734 ± 0.060 | 0.884 ± 0.018 | 0.805 ± 0.028 | best FS: `MW-inCV`; top feats: hydrophobic_janin, hydrophobic_eisenberg, X5_K |
| KNN (UOOP best FS) | 0.732 ± 0.060 | 0.815 ± 0.025 | 0.713 ± 0.034 | best FS: `Chi-square-inCV`; top feats: hydrophobic_janin, hydrophobic_kyte-doolittle, hydrophobic_eisenberg |
| Decision Tree (UOOP best FS) | 0.726 ± 0.051 | 0.844 ± 0.037 | 0.751 ± 0.049 | best FS: `Kruskal-inCV`; top feats: hydrophobic_janin, hydrophobic_kyte-doolittle, X5_K |
| Majority-class baseline (always predict 1) | 0.500 | 0.927 | 0.863 | Reference floor for AUC; ceiling for F1/ACC on this imbalance |

> **UOOP source:** `UOOP-Project/results/smote_in_cv/20260523_144512/feature_selection_metrics_20260523_144525.csv`.
> Each row aggregates 100 fits (10-fold × 10 repeats), so std columns are tighter than Multi_CycGT's pure 10-fold std.
> Best FS picked post-hoc by max mean AUC per classifier — mildly optimistic; Multi_CycGT's early-stopped number needs no such post-hoc pick, so the comparison is conservative on UOOP's side.

## Per-fold breakdown (early-stopped / stop epoch)

| Fold | Stop epoch | Best-val-loss epoch | Val AUC | Test AUC | Test F1 | Test ACC |
|---|---|---|---|---|---|---|
| 1 | 57 | 7 | 0.673 | 0.748 | 0.843 | 0.750 |
| 2 | 53 | 3 | 0.644 | 0.713 | 0.882 | 0.797 |
| 3 | 68 | 18 | 0.803 | 0.817 | 0.874 | 0.789 |
| 4 | 55 | 5 | 0.671 | 0.786 | 0.811 | 0.719 |
| 5 | 56 | 6 | 0.713 | 0.791 | 0.828 | 0.734 |
| 6 | 52 | 2 | 0.788 | 0.782 | 0.828 | 0.734 |
| 7 | 58 | 8 | 0.737 | 0.636 | 0.765 | 0.641 |
| 8 | 60 | 10 | 0.873 | 0.736 | 0.647 | 0.539 |
| 9 | 67 | 17 | 0.849 | 0.762 | 0.776 | 0.680 |
| 10 | 53 | 3 | 0.780 | 0.795 | 0.887 | 0.812 |
| **mean ± std** | — | — | 0.753 ± 0.078 | **0.757 ± 0.052** | 0.814 ± 0.072 | 0.720 ± 0.083 |

## Observations

**Multimodal representation reaches parity with handcrafted-feature Random
Forest, but does not beat it.** Multi_CycGT's early-stopped test AUC of
**0.757 ± 0.052** is statistically indistinguishable from UOOP's best classical
baseline — Random Forest on all 233 handcrafted features at **0.764 ± 0.058**.
Every UOOP classifier with proper feature selection lands in the 0.73–0.76 AUC
band; the multimodal model joins that band rather than exceeding it. On F1 the
early-stopped model (0.814) is now much closer to RF (0.904) than the old
best-val run was (0.749), because the stop-epoch threshold is better behaved —
but RF still leads on F1/accuracy, partly because those metrics reward the
always-predict-positive bias on this 87/13 imbalance. AUC is the
apples-to-apples number, and on it **the rich multimodal representation is not
adding value over RF on handcrafted features** for this task at N=1771.

**Why no uplift over RF?** Three plausible explanations, in roughly decreasing
likelihood: (1) **Data scale.** 1771 samples is small for a ~2M-parameter
multimodal model; classical models with informative engineered features are
near-optimal on a dataset this size. (2) **Inductive-bias mismatch.** The graph
+ tokenized-SMILES + LogP/TPSA inputs were chosen to capture permeability
(lipophilicity, membrane interaction); synthesizability failures (aggregation,
β-sheet, difficult couplings) may correlate more with hydrophobicity *patterns*
across the sequence — which RF's `hydrophobic_janin`,
`hydrophobic_kyte-doolittle`, `acidic_group`, `peptide_len` features encode
directly. (3) **Capacity vs. regularization.** Even with early stopping the
fused head `fc1 = Linear(16552, 1)` is a large, lightly-regularized bottleneck;
dropout on the fused head or a lower LR might help, but as configured it reaches
parity, not dominance.

**Overfitting was the dominant failure mode — early stopping now contains it.**
The original 500-epoch run overfit hard (8 of 10 folds peaked by epoch ~22, then
degraded). Early stopping on val loss with patience 50 now halts each fold at
epoch 52–68 instead of 500 — a ~7–8× reduction in epochs — and the kept model
scores AUC 0.757 vs the old overfit final-epoch 0.661. This closes the gap that
previously required post-hoc epoch selection. Remaining headroom, if any, would
come from stronger regularization (dropout on the fused head, lower LR, or focal
loss in place of WeightedRandomSampler for a cleaner minority-class gradient).

**Calibration is still poor — predictions are bimodal and overconfident.** At
the stop epoch, 38% of pooled test predictions land in [0.9, 1.0) and ~10% in
[0.0, 0.1); only ~5% fall in the [0.4, 0.5) "uncertain" bin. Early stopping
improved ranking but not calibration (expected — keep-last sits past the val-loss
minimum, where confidence has already saturated). If you use the model for
**ranking** (prioritizing peptides for wet-lab attempt), AUC is the right metric
and 0.757 is the number to quote. If you need **calibrated probabilities** (risk
scores for triage), add temperature scaling or Platt calibration on a held-out
set.

**Cross-task transfer caveat (genuine, not a hedge).** Multi_CycGT was designed
for cyclic peptides + membrane permeability. We apply the same architecture to
linear peptides + synthesis success — different chemistry, different failure
modes (aggregation, β-sheet propensity, difficult couplings). The achieved AUC
~0.76 is consistent with "architecture is informative but inductive biases are
not perfectly aligned with the task."

## Hardware and runtime

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (Ampere, compute capability 8.6)
- torch 2.0.1+cu117, dgl 1.1.0+cu117, dgllife 0.3.2 (patched), Python 3.10 venv
- 10 folds × 52–68 early-stopped epochs (vs the 500-epoch ceiling) — roughly
  7–8× fewer epochs than the original un-early-stopped run, so wall-clock is
  correspondingly shorter.

## Reproducing the numbers

```powershell
# Early-stopped per-fold metrics (written by model_concat.py during training)
Get-Content model/deep_learning/gcn_transformer_fc/results_per_fold.csv

# Aggregate (per-fold + pooled), run from repo root:
python analyze_results.py

# Post-hoc best-val-AUC epoch comparison (writes results_per_fold_best.csv):
python analyze_best_epoch.py
```

`analyze_results.py` and `analyze_best_epoch.py` live at the repo root and
auto-discover each fold's final (stop) epoch from the prediction CSVs, so they
work even though folds now stop at different epochs.
