"""Post-hoc plots for the Multi_CycGT synthesizability run.

Reads artifacts already produced by model_concat.py (no re-training):
  - training_log.txt           -> per-epoch train/val/test loss (loss curves)
  - pred_data_origin/.../*.csv  -> per-epoch val/test AUC + ROC (auc/roc curves)
  - results_per_fold.csv        -> stop epoch + best-val-loss epoch markers

matplotlib is imported lazily inside the plot functions so the parser and the
prediction readers stay unit-testable without a matplotlib install.

Run from the repo root:  python plot_results.py
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


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_loss_curves(history, results, out_dir):
    plt = _pyplot()
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
    fig.suptitle("Loss vs epoch (train/val/test) - early-stopped", fontsize=14)
    fig.tight_layout()
    out = out_dir / "loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_auc_curves(results, out_dir):
    plt = _pyplot()
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
    plt = _pyplot()
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
    ax.set_title("ROC at stop epoch - 10 folds + mean")
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
