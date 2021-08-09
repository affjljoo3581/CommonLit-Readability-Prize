import itertools
from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import T5Config, T5Model, T5PreTrainedModel
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)
from modeling.mixins import ClassifierMixin, TransformerMixin


class T5ClassificationHead(AttentionBasedClassificationHead):
    """
    A classification head for T5 model. This class inherits
    `AttentionBasedClassificationHead` class and automatically finds necessary
    hyperparameters from `T5Config`.

    Args:
        config: The dictionary-based configuration object (`T5Config`).
    """

    def __init__(self, config: T5Config):
        super().__init__(config.d_model, config.num_labels, config.dropout_rate)


class T5ForCustomSequenceClassification(
    TransformerMixin,
    ClassifierMixin,
    T5PreTrainedModel,
):
    """A custom sequence classification model with T5.

    Compared to the original version (`T5ForSequenceClassification`), this model uses
    `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and t5 model.

    Args:
        config: The dictionary-based configuration object (`T5Config`).
    """

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.config = config

        self.transformer = T5Model(config)
        self.classifier = T5ClassificationHead(config)
        self.criterion = SequenceClassificationLoss(config)

        self.init_weights()

    def _init_weights(self, module: nn.Module):
        if module in self.classifier.modules() and isinstance(module, nn.Linear):
            module.weight.data.normal_(std=self.config.initializer_factor * 0.02)
            module.bias.data.zero_()
        else:
            super()._init_weights(module)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[Tuple[torch.Tensor, ...], Seq2SeqSequenceClassifierOutput]:
        return_dict = return_dict or self.config.use_return_dict

        outputs = self.transformer(
            input_ids=input_ids,
            decoder_input_ids=self.transformer._shift_right(input_ids),
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
        encoder_layers = [b.modules() for b in self.transformer.encoder.block[:-1]]
        encoder_layers.append(
            itertools.chain(
                self.transformer.encoder.block[-1].modules(),
                self.transformer.encoder.final_layer_norm.modules(),
            )
        )

        decoder_layers = [b.modules() for b in self.transformer.decoder.block[:-1]]
        decoder_layers.append(
            itertools.chain(
                self.transformer.decoder.block[-1].modules(),
                self.transformer.decoder.final_layer_norm.modules(),
            )
        )

        layers = encoder_layers + decoder_layers
        layers = layers[:num_layers] if not reverse else layers[: -num_layers - 1 : -1]

        for layer in layers:
            for module in layer:
                self._init_weights(module)

    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        yield self.transformer.shared.modules()

        yield from [b.modules() for b in self.transformer.encoder.block[:-1]]
        yield itertools.chain(
            self.transformer.encoder.block[-1].modules(),
            self.transformer.encoder.final_layer_norm.modules(),
        )

        yield from [b.modules() for b in self.transformer.decoder.block[:-1]]
        yield itertools.chain(
            self.transformer.decoder.block[-1].modules(),
            self.transformer.decoder.final_layer_norm.modules(),
        )

        yield self.classifier.modules()
