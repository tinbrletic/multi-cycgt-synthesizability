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
