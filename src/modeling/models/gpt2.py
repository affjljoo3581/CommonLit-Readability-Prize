import itertools
from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)
from modeling.mixins import ClassifierMixin, TransformerMixin


class GPT2ClassificationHead(AttentionBasedClassificationHead):
    """
    A classification head for GPT2 model. This class inherits
    `AttentionBasedClassificationHead` class and automatically finds necessary
    hyperparameters from `GPT2Config`.

    Args:
        config: The dictionary-based configuration object (`GPT2Config`).
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config.n_embd, config.num_labels, config.summary_first_dropout)


class GPT2ForCustomSequenceClassification(
    TransformerMixin,
    ClassifierMixin,
    GPT2PreTrainedModel,
):
    """A custom sequence classification model with GPT2.

    Compared to the original version (`GPT2ForSequenceClassification`), this model uses
    `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and gpt2 model.

    Args:
        config: The dictionary-based configuration object (`GPT2Config`).
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.config = config

        self.transformer = GPT2Model(config)
        self.classifier = GPT2ClassificationHead(config)
        self.criterion = SequenceClassificationLoss(config)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
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
            return ((loss, logits) if loss is not None else (logits,)) + outputs[1:]

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
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
        layers = [layer.modules() for layer in self.transformer.h]
        layers[-1] = itertools.chain(layers[-1], self.transformer.ln_f.modules())

        layers = layers[:num_layers] if not reverse else layers[: -num_layers - 1 : -1]
        for modules in layers:
            for module in modules:
                self._init_weights(module)

    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        yield itertools.chain(
            self.transformer.wte.modules(), self.transformer.wpe.modules()
        )
        yield from [layer.modules() for layer in self.transformer.h[:-1]]
        yield itertools.chain(
            self.transformer.h[-1].modules(), self.transformer.ln_f.modules()
        )
        yield self.classifier.modules()
