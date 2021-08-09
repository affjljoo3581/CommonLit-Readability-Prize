from typing import Any, Dict, Iterator

import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5LayerNorm

from modeling import TransformerMixin


def _get_do_decay_params(modules: Iterator[nn.Module]) -> Iterator[nn.Parameter]:
    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if not isinstance(module, (nn.LayerNorm, T5LayerNorm)) and name != "bias":
                yield param


def _get_no_decay_params(modules: Iterator[nn.Module]) -> Iterator[nn.Parameter]:
    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if isinstance(module, (nn.LayerNorm, T5LayerNorm)) or name == "bias":
                yield param


def create_param_groups(
    model: TransformerMixin,
    learning_rate: float,
    layerwise_lr_decay: float = 1.0,
    weight_decay: float = 0.01,
) -> Iterator[Dict[str, Any]]:
    """
    Create parameter groups which have different learning rate and weight decay rate.

    Usually weight decay is not performed to `LayerNorm` parameters and biases. To
    ignore them from weight decaying, it is required to specify different weight decay
    rate to each parameter group.

    This function supports not only weight decay separation but layer-wise learning rate
    decaying. It is well-known to apply different learning rate to each transformer
    layer to get better performance. This function iterates the reversed hierarchical
    transformer layers by using `TransformerMixin` mixin, and assign different learning
    rate by multiplying the decay rate to the previous one.

    Examples::

        >>> model: TransformerMixin = ...
        >>> optimizer = AdamW(
        >>>     create_param_groups(
        >>>         model,
        >>>         learning_rate=5e-5,
        >>>         layerwise_lr_decay=0.95,
        >>>         weight_decay=0.01
        >>>     ),
        >>>     betas=(0.9, 0.98),
        >>>     eps=1e-6,
        >>> )

    Args:
        model: The transformer model which inherits `TransformerMixin` mixin. As
            mentioned above, this function implements layer-wise learning rate decaying
            by using the mixin. Therefore it needs to pass the model with mixin.
        learning_rate: The base learning rate. It is also the maximum learning rate
            among the layers.
        layerwise_lr_decay: The rate of layer-wise learning rate decay. Default is
            `0.1`.
        weight_decay: The rate of weight decay. As mentioned above, `LayerNorm`
            parameters and biases will not be decayed and the weight decay rate of them
            is explicitly set to `0`. Default is `0.01`.

    Yields:
        A parameter group containing learning rate and weight decay rate.
    """
    for i, modules in enumerate(reversed(list(model.hierarchical_modules()))):
        # While the generator becomes empty after consuming all data, we need to copy
        # the generator of modules to use both do-decay and no-decay filtering.
        modules = list(modules)

        yield {
            "params": _get_do_decay_params(modules),
            "lr": learning_rate * layerwise_lr_decay ** i,
            "weight_decay": weight_decay,
        }
        yield {
            "params": _get_no_decay_params(modules),
            "lr": learning_rate * layerwise_lr_decay ** i,
            "weight_decay": 0.0,
        }
