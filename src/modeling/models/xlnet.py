from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import XLNetConfig, XLNetModel, XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import (
    XLNetForSequenceClassificationOutput,
)

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)
from modeling.mixins import ClassifierMixin, TransformerMixin


class XLNetClassificationHead(AttentionBasedClassificationHead):
    """
    A classification head for XLNet model. This class inherits
    `AttentionBasedClassificationHead` class and automatically finds necessary
    hyperparameters from `XLNetConfig`.

    Args:
        config: The dictionary-based configuration object (`XLNetConfig`).
    """

    def __init__(self, config: XLNetConfig):
        super().__init__(config.d_model, config.num_labels, config.dropout)


class XLNetForCustomSequenceClassification(
    TransformerMixin,
    ClassifierMixin,
    XLNetPreTrainedModel,
):
    """A custom sequence classification model with XLNet.

    Compared to the original version (`XLNetForSequenceClassification`), this model uses
    `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and bert model.

    Args:
        config: The dictionary-based configuration object (`XLNetConfig`).
    """

    def __init__(self, config: XLNetConfig):
        super().__init__(config)
        self.config = config

        self.transformer = XLNetModel(config)
        self.classifier = XLNetClassificationHead(config)
        self.criterion = SequenceClassificationLoss(config)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[Tuple[torch.Tensor, ...], XLNetForSequenceClassificationOutput]:
        return_dict = return_dict or self.config.use_return_dict

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        logits = self.classifier(outputs[0], attention_mask=attention_mask)
        loss = self.criterion(logits, labels) if labels is not None else None

        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,)) + outputs[2:]

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
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
            target_layers = self.transformer.layer[: -num_layers - 1 : -1]
        else:
            target_layers = self.transformer.layer[:num_layers]

        for layer in target_layers:
            for module in layer.modules():
                self._init_weights(module)

    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        yield self.transformer.word_embedding.modules()
        yield from [layer.modules() for layer in self.transformer.layer]
        yield self.classifier.modules()
