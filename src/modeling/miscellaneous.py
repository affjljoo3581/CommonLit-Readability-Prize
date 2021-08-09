from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class SequenceClassificationLoss(nn.Module):
    """Calculate a loss for sequence classification.

    Using transformer models, you can classify sentences to single-label category,
    multi-label classes, and even do regression. This class supports various cases of
    the sequence classification to calculate losses. Basically, the problem type is
    determined by `num_labels` and a type of `labels` passed by input arguments. For
    example, when the model has single label, then the task is inferred to regression.

    The internal implementation of this class is brought from
    `*ForSequenceClassification` models in `transformers` library. While the loss
    calculation parts are duplicated, this class is separately written to reuse to new
    custom classification models.

    Args:
        config: A dictionary-based configuration object (`PretrainedConfig`). A
            necessary information for determining the problem type and calculating loss
            is already defined at `PretrainedConfig` class, there is no need to
            implement different versions of this class according to the configuration
            types. You can simply pass any type of configurations which inherit
            `PretrainedConfig` class.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.config.problem_type is None:
            # Update the problem type with the number of labels when the type is not
            # defined.
            if self.config.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.config.num_labels > 1:
                if labels.dtype == torch.long or labels.dtype == torch.int:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

        # Calculate the classification loss.
        if self.config.problem_type == "regression":
            if self.config.num_labels == 1:
                logits = logits.squeeze()
                labels = labels.squeeze()
            return F.mse_loss(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            logits = logits.view(-1, self.config.num_labels)
            labels = labels.flatten()
            return F.cross_entropy(logits, labels)
        elif self.config.problem_type == "multi_label_classification":
            return F.binary_cross_entropy_with_logits(logits, labels)


class AttentionBasedClassificationHead(nn.Module):
    """An attention-based classification head.

    This class is used to pool the hidden states from transformer model and computes the
    logits. In order to calculate the worthful representations from the transformer
    outputs, this class adopts time-based attention gating. Precisely, this class
    computes the importance of each word by using features and then apply
    weight-averaging to the features across the time axis.

    Since original classification models (i.e. `*ForSequenceClassification`) use simple
    feed-forward layers, the attention-based classification head can learn better
    generality.

    Args:
        hidden_size: The dimensionality of hidden units.
        num_labels: The number of labels to predict.
        dropout_prob: The dropout probability used in both attention and projection
            layers. Default is `0.1`.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Calculate the attention scores and apply the attention mask so that the
        # features of the padding tokens are not attended to the representation.
        attn = self.attention(features)
        if attention_mask is not None:
            attn += (1 - attention_mask.unsqueeze(-1)) * -10000.0

        # Pool the features across the timesteps and calculate logits.
        x = (features * attn.softmax(dim=1)).sum(dim=1)
        return self.classifier(x)
