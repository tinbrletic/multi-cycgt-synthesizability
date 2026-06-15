"""Early-stopping helper for the Multi_CycGT training loop.

Pure logic, no torch/CUDA dependency, so it can be unit-tested in isolation.
Tracks a monitored validation metric across epochs and decides when training
should halt after `patience` epochs without improvement.

Used by model_concat.py with mode="min" on validation loss. The mode="max"
path (e.g. validation AUC) is supported for future reuse.
"""


class EarlyStopping:
    """Decide when to stop training based on a monitored metric.

    Call :meth:`step` once per epoch with the metric value. It returns True the
    moment training should stop (``num_bad_epochs`` reached ``patience``).

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without a qualifying improvement that are
        tolerated before stopping.
    min_delta : float
        Minimum change over the current best that counts as an improvement. A
        change smaller than this is treated as no improvement (and does not
        update ``best``).
    mode : {"min", "max"}
        "min" if a lower metric is better (e.g. loss), "max" if higher is
        better (e.g. AUC).

    Attributes
    ----------
    best : float | None
        Best metric value seen so far (None before the first step).
    best_epoch : int
        1-based epoch index at which ``best`` was achieved.
    num_bad_epochs : int
        Epochs since the last qualifying improvement.
    should_stop : bool
        True once a stop has been signalled.
    """

    def __init__(self, patience=50, min_delta=0.0, mode="min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = None
        self.best_epoch = 0
        self.num_bad_epochs = 0
        self.should_stop = False
        self._epoch = 0

    def _is_improvement(self, value):
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value):
        """Record this epoch's metric; return True if training should stop."""
        self._epoch += 1

        if self.best is None or self._is_improvement(value):
            self.best = value
            self.best_epoch = self._epoch
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop
