import pytorch_lightning as pl


class ValidationEpochLogger(pl.Callback):
    """Log validation epochs with considering the validation checking interval.

    When you are using pytorch-lightning with `val_check_interval` parameter, the
    validation loops will be performed within the training epochs. Since
    pytorch-lightning logs the validation metrics with discretized epochs or global
    steps, it is hard to normalize when the batch size is different to each other.
    Hence, this class logs the splitted validation epochs after the end of every
    validation epochs.

    Args:
        name: The metric name of the logged validation epochs. Default is `val/epoch`.
        last_epoch: The last validation epoch. Default is `-1`.
    """

    def __init__(self, name: str = "val/epoch", last_epoch: int = -1):
        super().__init__()
        self.name = name
        self.val_epoch = last_epoch

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not trainer.sanity_checking and not trainer.should_stop:
            self.val_epoch += 1
            pl_module.log(self.name, self.val_epoch * trainer.val_check_interval)
