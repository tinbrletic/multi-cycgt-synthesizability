# Results Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A post-hoc `plot_results.py` that turns the completed early-stopped run's existing artifacts into three figures (loss curves, AUC curves, ROC curves) with no re-training.

**Architecture:** One repo-root script (sibling to `analyze_results.py`). A pure `parse_training_log` reads per-epoch losses from `training_log.txt`; `per_epoch_auc`/`final_epoch`/`load_preds` read the per-epoch prediction CSVs (same logic as `analyze_best_epoch.py`). Three `plot_*` functions emit PNGs to `model/deep_learning/gcn_transformer_fc/figures/`. matplotlib is imported lazily inside the plot functions so the parser is testable without it.

**Tech Stack:** Python 3.10, matplotlib (Agg backend), pandas, numpy, scikit-learn. Tests use a standalone `__main__` runner (no pytest), mirroring `early_stopping.py`.

---

## File structure

- Create: `plot_results.py` (repo root) — parser + readers + 3 plot functions + `main()`.
- Create: `tests/test_plot_results.py` (repo root) — unit tests for `parse_training_log` and `final_epoch`.
- Output (generated, not hand-written): `model/deep_learning/gcn_transformer_fc/figures/{loss_curves,auc_curves,roc_curves}.png`.

## Task 0: Install matplotlib into the venv

- [ ] **Step 1: Install**

Run: `.\.venv\Scripts\python.exe -m pip install matplotlib`

- [ ] **Step 2: Verify import**

Run: `.\.venv\Scripts\python.exe -c "import matplotlib; print(matplotlib.__version__)"`
Expected: a version string (e.g. `3.x.y`), no error.

## Task 1: `parse_training_log` (pure parser, TDD)

**Files:**
- Create: `plot_results.py`
- Test: `tests/test_plot_results.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_plot_results.py`:

```python
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_results import parse_training_log  # noqa: E402

SAMPLE = """74
epoch: 1, train_LOSS      : 0.900, train_ACC        : 0.50
epoch: 1, test_LOSS       : 0.800, test_ACC         : 0.60
epoch: 1, val_LOSS        : 1.100, val_ACC          : 0.55
epoch: 2, train_LOSS      : 0.700, train_ACC        : 0.65
epoch: 2, test_LOSS       : 0.750, test_ACC         : 0.62
epoch: 2, val_LOSS        : 1.050, val_ACC          : 0.58
Fold 1: early stopping at epoch 2 (best val loss 1.050 @ epoch 2; no improvement for 50 epochs)
Fold 1 final (epoch 2): test AUC=0.70 F1=0.8 ACC=0.6 on 128/178 samples
epoch: 1, train_LOSS      : 0.950, train_ACC        : 0.48
epoch: 1, test_LOSS       : 0.820, test_ACC         : 0.59
epoch: 1, val_LOSS        : 1.200, val_ACC          : 0.52
Fold 2 final (epoch 1): test AUC=0.71 F1=0.8 ACC=0.6 on 128/177 samples
"""


def _parse_sample():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE)
        path = f.name
    try:
        return parse_training_log(Path(path))
    finally:
        os.unlink(path)


def test_segments_two_folds():
    h = _parse_sample()
    assert set(h) == {1, 2}, list(h)


def test_epoch_numbering_resets_per_fold():
    h = _parse_sample()
    assert h[1]["epoch"] == [1, 2]
    assert h[2]["epoch"] == [1]


def test_skips_noise_lines():
    # the leading '74' and the 'early stopping' line must not create epochs
    h = _parse_sample()
    assert h[1]["train_loss"] == [0.9, 0.7]
    assert h[1]["val_loss"] == [1.1, 1.05]
    assert h[1]["test_loss"] == [0.8, 0.75]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe tests\test_plot_results.py`
Expected: ImportError / ModuleNotFoundError for `plot_results` (file not created yet).

- [ ] **Step 3: Write minimal implementation** — create `plot_results.py` with imports + parser:

```python
"""Post-hoc plots for the Multi_CycGT synthesizability run.

Reads artifacts already produced by model_concat.py (no re-training):
  - training_log.txt        -> per-epoch train/val/test loss  (loss curves)
  - pred_data_origin/.../*.csv -> per-epoch val/test AUC + ROC (auc/roc curves)
  - results_per_fold.csv    -> stop epoch + best-val-loss epoch markers

matplotlib is imported lazily inside the plot functions so the parser stays
unit-testable without a matplotlib install.
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

RESULTS_CSV = Path("model/deep_learning/gcn_transformer_fc/results_per_fold.csv")
PRED_DIR = Path("model/deep_learning/gcn_transformer_fc/pred_data_origin/gcn_transformer_fc")
LOG_PATH = Path("training_log.txt")
FIG_DIR = Path("model/deep_learning/gcn_transformer_fc/figures")

_EPOCH_RE = re.compile(
    r"epoch:\s*(\d+),\s*(train|test|val)_LOSS\s*:\s*([\d.]+),\s*\w+_ACC\s*:\s*([\d.]+)"
)
_FINAL_RE = re.compile(r"Fold\s+(\d+)\s+final")


def parse_training_log(path):
    """Parse the Tee'd training log into per-fold per-epoch loss/acc series.

    Returns {fold: {"epoch": [...], "train_loss": [...], "val_loss": [...],
    "test_loss": [...], "train_acc": [...], "val_acc": [...], "test_acc": [...]}}.
    Fold blocks are delimited by the "Fold N final" marker; epoch numbering
    resets per fold. The leading n_feats print ("74") and the "early stopping"
    lines do not match _EPOCH_RE and are ignored.
    """
    folds = {}
    cur = defaultdict(dict)  # epoch -> {"train_loss": x, "train_acc": y, ...}

    def flush(fold_idx):
        if not cur:
            return
        epochs = sorted(cur)
        rec = {"epoch": epochs}
        for split in ("train", "val", "test"):
            rec[f"{split}_loss"] = [cur[e].get(f"{split}_loss") for e in epochs]
            rec[f"{split}_acc"] = [cur[e].get(f"{split}_acc") for e in epochs]
        folds[fold_idx] = rec

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _EPOCH_RE.search(line)
            if m:
                ep, split = int(m.group(1)), m.group(2)
                cur[ep][f"{split}_loss"] = float(m.group(3))
                cur[ep][f"{split}_acc"] = float(m.group(4))
                continue
            fm = _FINAL_RE.search(line)
            if fm:
                flush(int(fm.group(1)))
                cur = defaultdict(dict)
    return folds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe tests\test_plot_results.py`
Expected: `3/3 passed`.

- [ ] **Step 5: Commit**

```bash
git add plot_results.py tests/test_plot_results.py
git commit -m "feat: training-log parser for plotting (TDD)"
```

## Task 2: prediction readers + `final_epoch` (TDD for discovery)

**Files:**
- Modify: `plot_results.py` (append functions)
- Test: `tests/test_plot_results.py` (append one test)

- [ ] **Step 1: Add the failing test** — append to `tests/test_plot_results.py` (before the `__main__` block):

```python
def test_final_epoch_discovers_max():
    import plot_results
    with tempfile.TemporaryDirectory() as d:
        td = Path(d) / "5" / "test"
        td.mkdir(parents=True)
        for ep in (1, 7, 3):
            (td / f"experiment_{ep}_predicted_test_values.csv").write_text("predict,true\n0.5,1\n")
        old = plot_results.PRED_DIR
        plot_results.PRED_DIR = Path(d)
        try:
            assert plot_results.final_epoch(5) == 7
        finally:
            plot_results.PRED_DIR = old
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe tests\test_plot_results.py`
Expected: `FAIL test_final_epoch_discovers_max` (AttributeError: no `final_epoch`).

- [ ] **Step 3: Implement** — append to `plot_results.py`:

```python
def load_preds(csv_path):
    """(preds, labels) for rows with a non-NaN predict; (None, None) if empty."""
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


def final_epoch(fold):
    """Largest epoch index with a test-prediction CSV for this fold (stop epoch)."""
    test_dir = PRED_DIR / str(fold) / "test"
    epochs = [
        int(p.stem.split("_")[1])
        for p in test_dir.glob("experiment_*_predicted_test_values.csv")
    ]
    return max(epochs) if epochs else None


def per_epoch_auc(fold, split):
    """{epoch: auc} across all per-epoch CSVs for a fold. split in {'test','val'}."""
    name = "test" if split == "test" else "valid"
    out = {}
    for p in (PRED_DIR / str(fold) / split).glob(f"experiment_*_predicted_{name}_values.csv"):
        ep = int(p.stem.split("_")[1])
        out[ep] = safe_auc(*load_preds(p))
    return dict(sorted(out.items()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe tests\test_plot_results.py`
Expected: `4/4 passed`.

- [ ] **Step 5: Commit**

```bash
git add plot_results.py tests/test_plot_results.py
git commit -m "feat: prediction-CSV readers + stop-epoch discovery"
```

## Task 3: plot functions + `main()`, validated by smoke run

**Files:**
- Modify: `plot_results.py` (append plot functions + `main()` + `__main__` guard)

- [ ] **Step 1: Implement** — append to `plot_results.py`:

```python
def _fig_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_loss_curves(history, results, out_dir):
    plt = _fig_axes()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, fold in zip(axes.ravel(), sorted(history)):
        h = history[fold]
        ax.plot(h["epoch"], h["train_loss"], label="train", color="tab:blue")
        ax.plot(h["epoch"], h["val_loss"], label="val", color="tab:orange")
        ax.plot(h["epoch"], h["test_loss"], label="test", color="tab:green")
        stop = int(results.loc[results.fold == fold, "epoch"].iloc[0])
        ax.axvline(stop, color="red", ls="--", lw=1, label=f"stop@{stop}")
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=7)
    fig.suptitle("Loss vs epoch (train/val/test) — early-stopped", fontsize=14)
    fig.tight_layout()
    out = out_dir / "loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_auc_curves(results, out_dir):
    plt = _fig_axes()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, fold in zip(axes.ravel(), range(1, 11)):
        val, test = per_epoch_auc(fold, "val"), per_epoch_auc(fold, "test")
        ax.plot(list(val), list(val.values()), label="val AUC", color="tab:orange")
        ax.plot(list(test), list(test.values()), label="test AUC", color="tab:green")
        row = results.loc[results.fold == fold].iloc[0]
        stop, best = int(row["epoch"]), int(row["best_val_loss_epoch"])
        ax.axvline(stop, color="red", ls="--", lw=1, label=f"stop@{stop}")
        ax.axvline(best, color="purple", ls=":", lw=1, label=f"bestval@{best}")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"Fold {fold}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=7)
    fig.suptitle("AUC vs epoch (val/test)", fontsize=14)
    fig.tight_layout()
    out = out_dir / "auc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_roc(out_dir):
    plt = _fig_axes()
    fig, ax = plt.subplots(figsize=(7, 7))
    mean_fpr = np.linspace(0, 1, 200)
    tprs, all_p, all_l = [], [], []
    for fold in range(1, 11):
        fe = final_epoch(fold)
        if fe is None:
            continue
        pr, la = load_preds(PRED_DIR / str(fold) / "test" / f"experiment_{fe}_predicted_test_values.csv")
        if pr is None or len(set(la.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(la, pr)
        ax.plot(fpr, tpr, lw=0.8, alpha=0.4, label=f"fold {fold} (AUC {auc(fpr, tpr):.2f})")
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        all_p.append(pr)
        all_l.append(la)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    pooled = roc_auc_score(np.concatenate(all_l), np.concatenate(all_p))
    ax.plot(mean_fpr, mean_tpr, color="navy", lw=2.5, label=f"mean ROC (pooled AUC {pooled:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC at stop epoch — 10 folds + mean")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = out_dir / "roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    results = pd.read_csv(RESULTS_CSV)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    history = parse_training_log(LOG_PATH)
    outs = [
        plot_loss_curves(history, results, FIG_DIR),
        plot_auc_curves(results, FIG_DIR),
        plot_roc(FIG_DIR),
    ]
    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on the real artifacts (from repo root)**

Run: `.\.venv\Scripts\python.exe plot_results.py`
Expected: three `wrote ...figures\*.png` lines, no traceback.

- [ ] **Step 3: Verify the three PNGs exist and are non-empty**

Run: `.\.venv\Scripts\python.exe -c "from pathlib import Path; d=Path('model/deep_learning/gcn_transformer_fc/figures'); [print(p.name, p.stat().st_size) for p in sorted(d.glob('*.png'))]"`
Expected: `auc_curves.png`, `loss_curves.png`, `roc_curves.png` each with size > 0.

- [ ] **Step 4: Re-run unit tests (guard against import regressions)**

Run: `.\.venv\Scripts\python.exe tests\test_plot_results.py`
Expected: `4/4 passed`.

- [ ] **Step 5: Commit**

```bash
git add plot_results.py model/deep_learning/gcn_transformer_fc/figures
git commit -m "feat: loss/AUC/ROC figures from existing run artifacts"
```

## Self-review

- **Spec coverage:** training-dynamics curves → Tasks 1+3 (`loss_curves.png`, `auc_curves.png`); ROC → Task 3 (`roc_curves.png`); post-hoc/no-retrain → all readers consume existing files; matplotlib dependency → Task 0; parser unit-tested → Task 1; output dir `figures/` → `FIG_DIR`. All covered.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type consistency:** `parse_training_log` returns dict-of-dict-of-lists consumed by `plot_loss_curves`; `per_epoch_auc`/`final_epoch`/`load_preds`/`safe_auc` names match between definition and call sites; `PRED_DIR`/`RESULTS_CSV`/`LOG_PATH`/`FIG_DIR` module constants used consistently.
