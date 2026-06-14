# Results plotting — design spec

**Date:** 2026-06-14
**Status:** Approved (design A)
**Related:** [[2026-06-12-early-stopping-design]], `comparison.md`, `analyze_results.py`, `analyze_best_epoch.py`

## Problem

The Multi_CycGT synthesizability run produces numbers (per-fold CSVs, a text
training log) but no figures. For the thesis/mentor writeup we want a visual
view of (1) how the model learns over time and (2) the ROC-AUC results. Today
there is no plotting code in the pipeline, so nothing is generated.

## Goal

A post-hoc plotting script that turns the **artifacts already on disk** (from the
completed early-stopped run) into figures, with **no re-training**. Two figure
families were approved:

1. **Training-dynamics curves** — "how the model learns over time."
2. **ROC curves** — "visually see the ROC-AUC."

## Approach (A — post-hoc, no re-run)

A standalone script `plot_results.py` at the repo root (sibling to
`analyze_results.py` and `analyze_best_epoch.py`, run from the repo root) reads
the existing artifacts and writes PNG figures. It does **not** touch the training
loop and requires no re-run. Rejected alternatives: instrumenting
`model_concat.py` to log per-fold history (would force a full re-run to get
history for the current results, and couples plotting to training); a minimal
ROC-only variant (too sparse for the "over time" requirement).

## Data sources (all already on disk)

| Plot data | Source |
|---|---|
| Loss vs epoch (train/val/test) | `training_log.txt` (repo root) — per-epoch lines, fold-delimited by `Fold N final (epoch M)` markers |
| AUC vs epoch (val/test) | `model/deep_learning/gcn_transformer_fc/pred_data_origin/gcn_transformer_fc/{fold}/{test,val}/experiment_{epoch}_predicted_*_values.csv` (columns `predict`, `true`) |
| ROC curves | the stop-epoch prediction CSVs (same files, largest epoch per fold) |
| Stop / best-val-loss epoch markers | `results_per_fold.csv` (`epoch`, `best_val_loss_epoch`) |

## Components (designed for isolation + testability)

- **`parse_training_log(path)`** → `{fold: [PerEpoch, ...]}` where each entry
  holds `(epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc)`.
  Pure parser: segments the log into fold blocks at each `Fold N final` marker,
  skips the leading `74` (n_feats print) and the `early stopping` print lines,
  and resets epoch numbering per fold. No matplotlib/IO beyond reading the file.
- **`per_epoch_auc(pred_dir, fold, split)`** → `{epoch: auc}`. Reuses the
  `load_preds` / `safe_auc` logic established in `analyze_best_epoch.py`
  (drop NaN `predict`, AUC only when both classes present).
- **`final_epoch(pred_dir, fold)`** → largest epoch with a test-prediction CSV
  (same discovery helper already used by the analysis scripts).
- **`plot_loss_curves(history, results, out_dir)`** → `loss_curves.png`.
- **`plot_auc_curves(pred_dir, results, out_dir)`** → `auc_curves.png`.
- **`plot_roc(pred_dir, results, out_dir)`** → `roc_curves.png`.
- **`main()`** — orchestrates: load `results_per_fold.csv`, parse the log, build
  the three figures.

## Figures

Output directory: `model/deep_learning/gcn_transformer_fc/figures/`
(co-located with `comparison.md`). matplotlib only (no seaborn), ~150 dpi.

1. **`loss_curves.png`** — 2×5 grid, one panel per fold. train/val/test loss vs
   epoch; vertical line at the stop epoch. Shows the train↓ / val↑ overfitting
   split that early stopping cuts off.
2. **`auc_curves.png`** — 2×5 grid, one panel per fold. val + test AUC vs epoch;
   markers at the best-val-loss epoch and the stop epoch.
3. **`roc_curves.png`** — single panel. 10 thin per-fold ROC curves at their stop
   epochs + a bold mean ROC + the chance diagonal; legend reports the pooled AUC
   (≈0.751).

## Testing

- `parse_training_log` is pure logic → unit tests with a standalone `__main__`
  runner so they run under plain `python` (no pytest dependency), mirroring
  `early_stopping.py` / `tests/test_early_stopping.py`. Cases: a two-fold
  synthetic log segments correctly; epoch numbering resets per fold; the leading
  `74` line and `early stopping` lines are ignored; malformed/blank lines are
  skipped.
- Plot functions are validated by a smoke run: invoke `main()` on the real
  artifacts and assert the three PNGs exist and are non-empty. (No pixel-level
  assertions.)

## Dependencies

- Add `matplotlib` to the venv (`pip install matplotlib`). Currently missing;
  sklearn/pandas/numpy are present.

## Out of scope

- Per-fold metric bar chart and calibration histogram (deferred; not in this
  spec's approved scope).
- Instrumenting `model_concat.py` or any re-training.
- Interactive/HTML plots; seaborn styling.
- Pinning matplotlib in `requirements.txt` (consistent with the repo's existing
  practice of leaving torch/rdkit/sklearn unpinned there).
