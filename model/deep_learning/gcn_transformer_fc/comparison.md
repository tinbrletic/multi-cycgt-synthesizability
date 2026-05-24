# Multi_CycGT GCN-Transformer-FC vs UOOP Classical Baselines

> **Status:** template. Fill in the rows below from `results_per_fold.csv`
> (this branch's output) and from your UOOP-Project's
> `results/smote_in_cv/<timestamp>/feature_selection_metrics_*.csv`.

## Dataset

1771 linear peptides (Gutman et al. 2022), binary synthesis success label,
**87/13 class imbalance** (1539 positive / 232 negative).

## Evaluation protocols

| Pipeline | CV | Imbalance handling | Hyperparameters |
|---|---|---|---|
| Multi_CycGT (this branch) | `KFold(n_splits=10, shuffle=True, random_state=3407)` | `WeightedRandomSampler` inside training DataLoader (no leakage — only on `X_train{i}.csv`) | 500 epochs (upstream default; lower if overfitting), `batch_size=128`, Adam lr=1e-3, BCELoss after sigmoid |
| UOOP classical | `RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)` | SMOTE inside each train fold | Per-classifier; see UOOP-Project |

**Comparability note.** The two pipelines use different CV splits and
different random states, so per-fold scores are NOT paired. Compare only at
the mean ± std level. (A paired comparison would require modifying
Multi_CycGT's `data_processing.py` to consume external split files —
larger change, out of scope here.)

## Headline table

Replace `<fill>` from the respective per-fold metric CSVs. For UOOP
classifiers, pick the best feature-selection strategy per classifier (by mean
test AUC) and note which strategy in the Notes column.

| Model | Mean AUC ± std | Mean F1 ± std | Mean ACC ± std | Notes |
|---|---|---|---|---|
| **Multi_CycGT (gcn_transformer_fc)** | `<fill>` | `<fill>` | `<fill>` | WeightedRandomSampler, 500 epochs final-epoch metrics, ~128/177 test samples per fold (drop_last) |
| LogReg (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| Random Forest (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| SVM (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| KNN (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| Decision Tree (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |
| Naive Bayes (UOOP best FS) | `<fill>` | `<fill>` | `<fill>` | best FS: `<which?>` |

## Per-fold breakdown

Optional but recommended for the final write-up. Copy from
`results_per_fold.csv`:

| Fold | Test AUC | Test F1 | Test ACC | Val AUC | n_test_eval / n_test_total |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| ... | | | | | |
| 10 | | | | | |

## Observations

(Write 2-3 paragraphs once results are in. Suggested angles:)

- **Did Multi_CycGT beat any classical baseline?** If yes, which and by how
  much (effect size — Cliff's delta or simple AUC-diff). If no, where does the
  classical pipeline win (precision/recall trade-off, calibration, runtime)?
- **Multimodal contribution.** Does the architecture's three-branch design
  show its value at N=1771, or does the small sample size favor classical
  classifiers with hand-crafted features? Reference the loss curve from
  `training_log.txt` — early plateau suggests underfitting (architecture
  starved for data); late plateau with diverging train/val loss suggests
  overfitting.
- **Cross-task transfer caveat.** Multi_CycGT was designed for cyclic
  peptides + permeability. Linear peptides + synthesizability is a different
  chemistry — permeability is governed by lipophilicity/TPSA/H-bond patterns;
  synthesis failure is governed by aggregation, beta-sheet propensity,
  difficult couplings. The architecture's representational biases may not
  transfer; an AUC of 0.6-0.7 would be a defensible academic finding, not a
  modeling failure.

## Hardware and runtime

- `<CPU or GPU model>`
- `<Total training time>`
- `<Per-fold training time>`
