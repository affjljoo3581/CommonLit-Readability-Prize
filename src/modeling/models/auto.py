from collections import OrderedDict

from transformers import (
    AlbertConfig,
    BartConfig,
    BertConfig,
    DebertaConfig,
    DebertaV2Config,
    DistilBertConfig,
    ElectraConfig,
    FunnelConfig,
    GPT2Config,
    MegatronBertConfig,
    MPNetConfig,
    RobertaConfig,
    T5Config,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
)
from transformers.models.auto.auto_factory import auto_class_factory

from modeling.models.albert import AlbertForCustomSequenceClassification
from modeling.models.bart import BartForCustomSequenceClassification
from modeling.models.bert import BertForCustomSequenceClassification
from modeling.models.deberta import DebertaForCustomSequenceClassification
from modeling.models.deberta_v2 import DebertaV2ForCustomSequenceClassification
from modeling.models.distilbert import DistilBertForCustomSequenceClassification
from modeling.models.electra import ElectraForCustomSequenceClassification
from modeling.models.funnel import FunnelForCustomSequenceClassification
from modeling.models.gpt2 import GPT2ForCustomSequenceClassification
from modeling.models.megatron_bert import MegatronBertForCustomSequenceClassification
from modeling.models.mpnet import MPNetForCustomSequenceClassification
from modeling.models.roberta import RobertaForCustomSequenceClassification
from modeling.models.t5 import T5ForCustomSequenceClassification
from modeling.models.xlm import XLMForCustomSequenceClassification
from modeling.models.xlm_roberta import XLMRobertaForCustomSequenceClassification
from modeling.models.xlnet import XLNetForCustomSequenceClassification

MODEL_FOR_CUSTOM_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (AlbertConfig, AlbertForCustomSequenceClassification),
        (BartConfig, BartForCustomSequenceClassification),
        (BertConfig, BertForCustomSequenceClassification),
        (DebertaConfig, DebertaForCustomSequenceClassification),
        (DebertaV2Config, DebertaV2ForCustomSequenceClassification),
        (DistilBertConfig, DistilBertForCustomSequenceClassification),
        (ElectraConfig, ElectraForCustomSequenceClassification),
        (FunnelConfig, FunnelForCustomSequenceClassification),
        (GPT2Config, GPT2ForCustomSequenceClassification),
        (MegatronBertConfig, MegatronBertForCustomSequenceClassification),
        (MPNetConfig, MPNetForCustomSequenceClassification),
        (RobertaConfig, RobertaForCustomSequenceClassification),
        (T5Config, T5ForCustomSequenceClassification),
        (XLMConfig, XLMForCustomSequenceClassification),
        (XLMRobertaConfig, XLMRobertaForCustomSequenceClassification),
        (XLNetConfig, XLNetForCustomSequenceClassification),
    ]
)

AutoModelForCustomSequenceClassification = auto_class_factory(
    "AutoModelForCustomSequenceClassification",
    MODEL_FOR_CUSTOM_SEQUENCE_CLASSIFICATION_MAPPING,
)
