# Spec — Early Stopping for Multi_CycGT (gcn_transformer_fc)

**Date:** 2026-06-12
**Status:** Approved (approach A)
**Scope:** Add validation-loss-based early stopping to the `gcn_transformer_fc`
training loop so folds stop training before overfitting, instead of always
running the fixed 500 epochs and saving the overfit final model.

## Motivation

In the last run, validation/test AUC for most folds peaked early (median best
epoch ~14-15) then degraded as the model overfit (val loss climbed to 2.5-3.7 by
epoch 500). Training runs a fixed 500 epochs with no early stopping, so the model
that gets scored/saved is the overfit final-epoch one. Post-hoc best-epoch
selection recovers ~+0.10 AUC, confirming the headroom. Early stopping moves that
fix *into* training.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Monitored signal | **Validation loss** (minimize) |
| Patience | **50 epochs** without improvement |
| Weights kept on stop | **Last (stopped) epoch** — no best-weight restore |
| Max-epoch ceiling | **500** (safety ceiling; early stop normally fires first) |
| Structure | **Approach A** — a small, unit-tested `EarlyStopping` helper class |

### Known trade-off (accepted)

Monitoring val *loss* + keeping the *last* epoch + patience 50 means the saved
model sits ~50 epochs past the val-loss minimum, and val-loss-min is generally a
different epoch than val-AUC-max under class imbalance. This is still far better
than epoch 500. To keep the AUC-optimal report available, per-epoch prediction
CSVs continue to be written (so `analyze_best_epoch.py` still works), and the
results CSV logs both the stop epoch and the best-val-loss epoch.

## Components

### New: `model/deep_learning/gcn_transformer_fc/early_stopping.py`

`EarlyStopping` — pure logic, no torch/CUDA dependency.

- `__init__(self, patience=50, min_delta=0.0, mode="min")`
  - `mode="min"` (loss) or `"max"` (e.g. AUC, for future reuse).
- `step(self, value) -> bool`
  - Call once per epoch. Updates internal best; resets the bad-epoch counter on
    an improvement strictly better than `best` by at least `min_delta`,
    otherwise increments it. Returns `True` when the counter reaches `patience`
    (i.e. caller should stop), else `False`.
- Public attributes for logging: `best`, `best_epoch`, `num_bad_epochs`,
  `should_stop`. An internal epoch counter increments on each `step`.

Improvement rule:
- `mode="min"`: improvement iff `value < best - min_delta`.
- `mode="max"`: improvement iff `value > best + min_delta`.

### Modified: `model/deep_learning/gcn_transformer_fc/model_concat.py`

- Module-level tunables: `EARLY_STOP_PATIENCE = 50`, `EARLY_STOP_MIN_DELTA = 0.0`.
- Per fold, before the epoch loop: `stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode="min")`.
- `for epoch in range(1, 501)` unchanged (500 ceiling).
- After the val pass each epoch (`val_epoch_acc, val_epoch_loss, val_list = ...`):
  `if stopper.step(val_epoch_loss): print(stop message); break`.
- "Keep last epoch" needs no snapshot: the existing post-loop scoring block
  already uses the last computed `test_list` / `val_list`.
- Extend the `results_per_fold.csv` row with: `stopped_epoch`,
  `best_val_loss_epoch`, `best_val_loss`. Existing columns unchanged so
  `analyze_best_epoch.py` / `comparison.md` aggregation keep working.

### New: `model/deep_learning/gcn_transformer_fc/tests/test_early_stopping.py`

Pure-logic unit tests (no torch/CUDA), written first (TDD):

1. Monotonically improving sequence never stops.
2. A plateau of exactly `patience` epochs after the best triggers a stop.
3. A late improvement resets the bad-epoch counter (no premature stop).
4. `min_delta` is respected (tiny improvements below it do not reset).
5. `best` / `best_epoch` track the optimum correctly.
6. `mode="max"` path works symmetrically (guards future AUC reuse).

## Out of scope (this change)

- Plotting / ROC-AUC visualization (separate follow-up task).
- Restoring best-epoch weights (explicitly not chosen).
- Applying early stopping to other variants (`gcn/`, `lstm/`, …).
- Decoupling `batch_size` from model dims / the `drop_last` truncation.

## Verification

- `python -m pytest model/deep_learning/gcn_transformer_fc/tests/test_early_stopping.py -v` passes.
- A short smoke check that `model_concat.py` imports and `EarlyStopping`
  integrates (without running the full CUDA training).
