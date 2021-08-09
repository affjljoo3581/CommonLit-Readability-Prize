import pytorch_lightning as pl


class ExtendedLightningModule(pl.LightningModule):
    """An extended pytorch-lighting module.

    This class is exactly the same as `pytorch_lightning.LightningModule` except one
    additional property `num_training_steps`. Usually it is important to calculate the
    total number of training steps to create a learning rate scheduler which is
    performed on every steps (not epochs). Unfortunately, the vanila `LightningModule`
    does not have any methods or properties for calculating the total training steps.
    Hence, this class supports the calculation with considering gradient accumulation,
    multi-device environment. Inherit this class and use `num_training_steps` property
    to get total training steps.
    """

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
