import itertools
from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BartConfig, BartModel, BartPretrainedModel
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)
from modeling.mixins import ClassifierMixin, TransformerMixin


class BartClassificationHead(AttentionBasedClassificationHead):
    """
    A classification head for Bart model. This class inherits
    `AttentionBasedClassificationHead` class and automatically finds necessary
    hyperparameters from `BartConfig`.

    Args:
        config: The dictionary-based configuration object (`BartConfig`).
    """

    def __init__(self, config: BartConfig):
        super().__init__(config.d_model, config.num_labels, config.classifier_dropout)


class BartForCustomSequenceClassification(
    TransformerMixin,
    ClassifierMixin,
    BartPretrainedModel,
):
    """A custom sequence classification model with Bart.

    Compared to the original version (`BartForSequenceClassification`), this model uses
    `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and bert model.

    Args:
        config: The dictionary-based configuration object (`BartConfig`).
    """

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.config = config

        self.model = BartModel(config)
        self.classifier = BartClassificationHead(config)
        self.criterion = SequenceClassificationLoss(config)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[Tuple[torch.Tensor, ...], Seq2SeqSequenceClassifierOutput]:
        return_dict = return_dict or self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        logits = self.classifier(outputs[0], attention_mask=attention_mask)
        loss = self.criterion(logits, labels) if labels is not None else None

        if not return_dict:
            return ((loss, logits) if loss is not None else (logits,)) + outputs[2:]

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def set_classifier_dropout(self, dropout_prob: float):
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_prob

    def init_classifier(self):
        for module in self.classifier.modules():
            self._init_weights(module)

    def init_transformer_layers(self, num_layers: int, reverse: bool = True):
        layers = list(self.model.encoder.layers) + list(self.model.decoder.layers)
        layers = layers[:num_layers] if not reverse else layers[: -num_layers - 1 : -1]

        for layer in layers:
            for module in layer.modules():
                self._init_weights(module)

    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        yield itertools.chain(
            self.model.shared.modules(),
            self.model.encoder.embed_positions.modules(),
            self.model.encoder.layernorm_embedding.modules(),
            self.model.decoder.embed_positions.modules(),
            self.model.decoder.layernorm_embedding.modules(),
        )
        yield from [layer.modules() for layer in self.model.encoder.layers]
        yield from [layer.modules() for layer in self.model.decoder.layers]
        yield self.classifier.modules()
