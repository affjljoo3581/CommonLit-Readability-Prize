import pytorch_lightning as pl


class BestScoreCallback(pl.Callback):
    """Record best score while training.

    This class records best validation metric score by comparing to the previous best
    score. You can access the accumulated best score by `best_score` property.

    Args:
        monitor: The target metric name to monitor.
        mode: The accumulation mode for the metric value. Default is `min`.
    """

    def __init__(self, monitor: str, mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else -float("inf")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        current = trainer.callback_metrics.get(self.monitor).item()

        if self.mode == "min":
            self.best_score = min(current, self.best_score)
        elif self.mode == "max":
            self.best_score = max(current, self.best_score)
