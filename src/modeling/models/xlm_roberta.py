from transformers import XLMRobertaConfig

from modeling.models.roberta import RobertaForCustomSequenceClassification


class XLMRobertaForCustomSequenceClassification(RobertaForCustomSequenceClassification):
    """A custom sequence classification model with XLM-Roberta.

    Compared to the original version (`RobertaForSequenceClassification`), this model
    uses `AttentionBasedClassificationHead` to classify the sentences. Also this class
    inherits `TransformerMixin` and `ClassifierMixin` mixins, so you can simply access
    detailed options or do complicated modifications (e.g. change the classifier dropout
    probability, re-initializing a portion of the transformer layers) without deep
    understanding of the internal implementations of this and bert model.

    Args:
        config: The dictionary-based configuration object (`XLMRobertaConfig`).
    """

    config_class = XLMRobertaConfig
