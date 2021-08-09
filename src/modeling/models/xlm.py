import itertools
from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import XLMConfig, XLMModel, XLMPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)
from modeling.mixins import ClassifierMixin, TransformerMixin


class XLMClassificationHead(AttentionBasedClassificationHead):
    """
    A classification head for XLM model. This class inherits
    `AttentionBasedClassificationHead` class and automatically finds necessary
    hyperparameters from `XLMConfig`.

    Args:
        config: The dictionary-based configuration object (`XLMConfig`).
    """

    def __init__(self, config: XLMConfig):
        super().__init__(config.hidden_size, config.num_labels, config.dropout)


class XLMForCustomSequenceClassification(
    TransformerMixin,
    ClassifierMixin,
    XLMPreTrainedModel,
):
    """A custom sequence classification model with XLM.

    Compared to the original version (`XLMForSequenceClassification`), this model uses
    `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and xlm model.

    Args:
        config: The dictionary-based configuration object (`XLMConfig`).
    """

    def __init__(self, config: XLMConfig):
        super().__init__(config)
        self.config = config

        self.transformer = XLMModel(config)
        self.classifier = XLMClassificationHead(config)
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
        target_layers = []
        for attn, ln1, ff, ln2 in zip(
            self.transformer.attentions,
            self.transformer.layer_norm1,
            self.transformer.ffns,
            self.transformer.layer_norm2,
        ):
            target_layers.append(
                itertools.chain(
                    attn.modules(),
                    ln1.modules(),
                    ff.modules(),
                    ln2.modules(),
                )
            )

        if reverse:
            target_layers = target_layers[: -num_layers - 1 : -1]
        else:
            target_layers = target_layers[:num_layers]

        for modules in target_layers:
            for module in modules:
                self._init_weights(module)

    def hierarchical_modules(self) -> Iterator[Iterator[nn.Module]]:
        if hasattr(self.transformer, "lang_embeddings"):
            yield itertools.chain(
                self.transformer.position_embeddings.modules(),
                self.transformer.lang_embeddings.modules(),
                self.transformer.embeddings.modules(),
                self.transformer.layer_norm_emb.modules(),
            )
        else:
            yield itertools.chain(
                self.transformer.position_embeddings.modules(),
                self.transformer.embeddings.modules(),
                self.transformer.layer_norm_emb.modules(),
            )

        for attn, ln1, ff, ln2 in zip(
            self.transformer.attentions,
            self.transformer.layer_norm1,
            self.transformer.ffns,
            self.transformer.layer_norm2,
        ):
            yield itertools.chain(
                attn.modules(),
                ln1.modules(),
                ff.modules(),
                ln2.modules(),
            )

        yield self.classifier.modules()
