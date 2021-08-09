from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)
from modeling.mixins import ClassifierMixin, TransformerMixin


class BertClassificationHead(AttentionBasedClassificationHead):
    """
    A classification head for Bert model. This class inherits
    `AttentionBasedClassificationHead` class and automatically finds necessary
    hyperparameters from `BertConfig`.

    Args:
        config: The dictionary-based configuration object (`BertConfig`).
    """

    def __init__(self, config: BertConfig):
        super().__init__(
            config.hidden_size, config.num_labels, config.hidden_dropout_prob
        )


class BertForCustomSequenceClassification(
    TransformerMixin,
    ClassifierMixin,
    BertPreTrainedModel,
):
    """A custom sequence classification model with Bert.

    Compared to the original version (`BertForSequenceClassification`), this model uses
    `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and bert model.

    Args:
        config: The dictionary-based configuration object (`BertConfig`).
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = BertClassificationHead(config)
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

        outputs = self.bert(
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
            target_layers = self.bert.encoder.layer[: -num_layers - 1 : -1]
        else:
            target_layers = self.bert.encoder.layer[:num_layers]

        for layer in target_layers:
            for module in layer.modules():
                self._init_weights(module)

    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        yield self.bert.embeddings.modules()
        yield from [layer.modules() for layer in self.bert.encoder.layer]
        yield self.classifier.modules()
