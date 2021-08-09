import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class LinearDecayLR(LRScheduler):
    """Decay the learning rate of each parameter groups linearly to zero.

    Args:
        optimizer: The target optimizer which contains the parameter group.
        total_steps: The number of total steps to decay.
        warmup_steps: The number of warmup steps. Default is `0`.
        last_epoch: The index of last epoch. Default is `-1`.
        verbose: The boolean that determines whether to print a message for each update
            or not. Default is `False`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use "
                "`get_last_lr()`."
            )

        decay_ratio = 1 - (self.last_epoch - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        warmup_ratio = self.last_epoch / self.warmup_steps if self.warmup_steps else 1

        return [base_lr * min(decay_ratio, warmup_ratio) for base_lr in self.base_lrs]
