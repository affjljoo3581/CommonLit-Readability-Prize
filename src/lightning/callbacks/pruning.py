import pytorch_lightning as pl
from optuna.trial import Trial


class OptunaPruningCallback(pl.Callback):
    """A PyTorch-Lightning callback for handling optuna's trial pruning.

    When you tune hyperparameters with optuna and pytorch lightning, you may use
    `PyTorchLightningPruningCallback` which is supported by optuna. However, the
    callback raises `TrialPruned` error and it occurs immediate training termination.
    This leads abnormal and unexpected results because the entire training procedures
    are not finalized completely. So rather than raising the error, this class
    explicitly set `should_stop` in pytorch-lightning trainer class to terminate the
    training normally. Note that it is also used in `EarlyStopping` in pytorch-lightning
    callbacks.

    Args:
        trial: The current trial which is used for reporting current metric and
            determining to be pruned.
        monitor: The target metric name to monitor.
        mode: The accumulation mode for the metric value. Default is `min`.
        report_best_score: The boolean whether to use best metric value or not. Default
            is `False`.
        last_epoch: The last validation epoch. Default is `-1`.
    """

    def __init__(
        self,
        trial: Trial,
        monitor: str,
        mode: str = "min",
        report_best_score: bool = False,
        last_epoch: int = -1,
    ):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
        self.report_best_score = report_best_score

        self.pruned = False
        self.val_epoch = last_epoch

        self.best_score = float("inf") if mode == "min" else -float("inf")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        self.val_epoch += 1

        if current_score is not None:
            # If `report_best_score==True`, use best score rather than current metric
            # value by comparing to the previous best metric value.
            if self.report_best_score:
                if self.mode == "min":
                    self.best_score = min(current_score, self.best_score)
                elif self.mode == "max":
                    self.best_score = max(current_score, self.best_score)
                current_score = self.best_score

            self.trial.report(current_score, step=self.val_epoch)
            self.pruned = trainer.should_stop = (
                trainer.should_stop or self.trial.should_prune()
            )
