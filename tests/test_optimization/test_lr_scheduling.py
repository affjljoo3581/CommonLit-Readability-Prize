import numpy as np
import pytest
import torch.nn as nn
from torch.optim import SGD

from optimization.lr_scheduling import LinearDecayLR


@pytest.mark.parametrize(
    "total_and_warmup_steps",
    [(10, 1), (100, 10), (100, 20), (1000, 10), (1000, 200), (1000, 500)],
)
def test_linear_decay_lr_learning_rates(total_and_warmup_steps):
    total_steps, warmup_steps = total_and_warmup_steps

    optimizer = SGD(nn.Linear(1, 1).parameters(), lr=1)
    scheduler = LinearDecayLR(optimizer, total_steps, warmup_steps)

    real_lrs = []
    for _ in range(total_steps):
        real_lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    steps = np.arange(total_steps)
    np_lrs = np.minimum(
        steps / warmup_steps, 1 - (steps - warmup_steps) / (total_steps - warmup_steps)
    )

    assert (np.array(real_lrs) == np_lrs).all()
