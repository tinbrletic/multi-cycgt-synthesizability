"""Unit tests for the EarlyStopping helper used by model_concat.py.

Pure-logic tests: no torch / CUDA / dataset needed. Runnable two ways:
    pytest model/deep_learning/gcn_transformer_fc/tests/test_early_stopping.py
    python model/deep_learning/gcn_transformer_fc/tests/test_early_stopping.py
(the __main__ block at the bottom runs every test_* function without pytest).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from early_stopping import EarlyStopping


def test_monotonically_improving_never_stops():
    """Loss that keeps falling should never trigger a stop."""
    stopper = EarlyStopping(patience=3, mode="min")
    for value in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        assert stopper.step(value) is False
    assert stopper.num_bad_epochs == 0
    assert stopper.best == 0.3
    assert stopper.best_epoch == 8


def test_plateau_of_patience_epochs_triggers_stop():
    """After the best, exactly `patience` non-improving epochs stops training."""
    stopper = EarlyStopping(patience=3, mode="min")
    assert stopper.step(0.5) is False   # epoch 1: new best
    assert stopper.step(0.6) is False   # epoch 2: bad #1
    assert stopper.step(0.6) is False   # epoch 3: bad #2
    assert stopper.step(0.6) is True    # epoch 4: bad #3 -> stop
    assert stopper.best_epoch == 1
    assert stopper.should_stop is True


def test_late_improvement_resets_counter():
    """An improvement before patience is exhausted must reset the bad counter."""
    stopper = EarlyStopping(patience=3, mode="min")
    stopper.step(0.5)                   # epoch 1: best
    stopper.step(0.6)                   # epoch 2: bad #1
    stopper.step(0.6)                   # epoch 3: bad #2
    assert stopper.step(0.4) is False   # epoch 4: new best -> reset
    assert stopper.num_bad_epochs == 0
    assert stopper.best == 0.4
    assert stopper.best_epoch == 4


def test_min_delta_is_respected():
    """Improvements smaller than min_delta do not count and do not reset."""
    stopper = EarlyStopping(patience=2, min_delta=0.1, mode="min")
    assert stopper.step(1.0) is False   # epoch 1: best
    assert stopper.step(0.95) is False  # epoch 2: only -0.05 (< 0.1) -> bad #1
    assert stopper.step(0.92) is True   # epoch 3: still < min_delta -> bad #2 -> stop
    assert stopper.best == 1.0          # best never updated below the delta


def test_best_and_best_epoch_track_optimum():
    """best / best_epoch follow the running optimum (min mode)."""
    stopper = EarlyStopping(patience=10, mode="min")
    for value in [0.5, 0.3, 0.4, 0.2, 0.6]:
        stopper.step(value)
    assert stopper.best == 0.2
    assert stopper.best_epoch == 4


def test_max_mode_is_symmetric():
    """mode='max' (e.g. AUC) stops when the metric stops rising."""
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.5) is False   # epoch 1: best
    assert stopper.step(0.6) is False   # epoch 2: new best
    assert stopper.step(0.55) is False  # epoch 3: bad #1
    assert stopper.step(0.54) is True   # epoch 4: bad #2 -> stop
    assert stopper.best == 0.6
    assert stopper.best_epoch == 2


if __name__ == "__main__":
    _tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    _failures = 0
    for _t in _tests:
        try:
            _t()
            print(f"PASS  {_t.__name__}")
        except AssertionError as exc:
            _failures += 1
            print(f"FAIL  {_t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            _failures += 1
            print(f"ERROR {_t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(_tests) - _failures}/{len(_tests)} passed")
    sys.exit(1 if _failures else 0)
