from abc import ABC, abstractmethod
from typing import Iterator

import torch.nn as nn


class TransformerMixin(ABC):
    """
    Mixin for transformer models. The classes which inherit this mixin provide the way
    to initialize weights in the transformer layers and hierarchically iterate the
    layers. Note that the implementation of the hierarchical iterating can be changed by
    tasks. For example, the yielded modules can be different when the model contains a
    sequence classification layer.
    """

    @abstractmethod
    def init_transformer_layers(self, num_layers: int, reverse: bool = True):
        """Initialize weights in a portion of transformer layers.

        Args:
            num_layers: The number of layers to initialize.
            reverse: The boolean whether to count the layers from bottom or top. Default
                is `True`.
        """
        ...

    @abstractmethod
    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        """
        Yield an iterator of each layers in transformer model. Note that details can be
        changed by the implementation of the model. For example, if the model contains
        embedding layer and projection layer for masked-lm, then they are also included
        and treated as one of the transformer layers. Therefore, you can access all
        parameters through this method, by following the order of hierarchical level.

        Examples::

            >>> param_groups = []
            >>> model: TransformerMixin = ...
            >>> for i, modules in enumerate(model.hierarchical_modules()):
            >>>     for module in modules:
            >>>         param_groups.append(
            >>>             {
            >>>                 "params": module.parameters(recurse=False),
            >>>                 "lr": 1e-4 * 0.95 ** (14 - i)
            >>>             }
            >>>         )

        Yields:
            An iterator over module parameters in each layer.
        """
        ...


class ClassifierMixin(ABC):
    """
    Mixin for classification models. The classes which inherit this mixin provide the
    way to modify the dropout probability at once and initialize the weights in the
    classification layer.
    """

    @abstractmethod
    def set_classifier_dropout(self, dropout_prob: float):
        """Change the dropout probability in the classification layer at once.

        Args:
            dropout_prob: The probability for dropout layers.
        """
        ...

    @abstractmethod
    def init_classifier(self):
        """Initialize weights in the classification layer."""
        ...
