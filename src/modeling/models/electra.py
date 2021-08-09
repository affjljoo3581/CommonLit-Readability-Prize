import itertools
from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import ElectraConfig, ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)
from modeling.mixins import ClassifierMixin, TransformerMixin


class ElectraClassificationHead(AttentionBasedClassificationHead):
    """
    A classification head for Electra model. This class inherits
    `AttentionBasedClassificationHead` class and automatically finds necessary
    hyperparameters from `ElectraConfig`.

    Args:
        config: The dictionary-based configuration object (`ElectraConfig`).
    """

    def __init__(self, config: ElectraConfig):
        super().__init__(
            config.hidden_size, config.num_labels, config.hidden_dropout_prob
        )


class ElectraForCustomSequenceClassification(
    TransformerMixin,
    ClassifierMixin,
    ElectraPreTrainedModel,
):
    """A custom sequence classification model with Electra.

    Compared to the original version (`ElectraForSequenceClassification`), this model
    uses `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and bert model.

    Args:
        config: The dictionary-based configuration object (`ElectraConfig`).
    """

    def __init__(self, config: ElectraConfig):
        super().__init__(config)
        self.config = config

        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)
        self.criterion = SequenceClassificationLoss(config)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        return_dict = return_dict or self.config.use_return_dict

        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        logits = self.classifier(outputs[0], attention_mask=attention_mask)
        loss = self.criterion(logits, labels) if labels is not None else None

        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,)) + outputs[2:]

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_classifier_dropout(self, dropout_prob: float):
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_prob

    def init_classifier(self):
        for module in self.classifier.modules():
            self._init_weights(module)

    def init_transformer_layers(self, num_layers: int, reverse: bool = True):
        if reverse:
            target_layers = self.electra.encoder.layer[: -num_layers - 1 : -1]
        else:
            target_layers = self.electra.encoder.layer[:num_layers]

        for layer in target_layers:
            for module in layer.modules():
                self._init_weights(module)

    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        if hasattr(self.electra, "embeddings_project"):
            yield itertools.chain(
                self.electra.embeddings.modules(),
                self.electra.embeddings_project.modules(),
            )
        else:
            yield self.electra.embeddings.modules()

        yield from [layer.modules() for layer in self.electra.encoder.layer]
        yield self.classifier.modules()
