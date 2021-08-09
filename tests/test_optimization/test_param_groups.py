import pytest
from transformers import AutoConfig

from modeling.models import AutoModelForCustomSequenceClassification
from optimization.param_groups import create_param_groups


@pytest.mark.parametrize(
    "name",
    [
        "albert-base-v2",
        "bert-base-cased",
        "roberta-base",
        "distilbert-base-cased",
        "google/electra-small-discriminator",
        "xlnet-base-cased",
        "facebook/bart-base",
        "microsoft/deberta-base",
        "funnel-transformer/small-base",
        "t5-small",
        "microsoft/mpnet-base",
        "distilgpt2",
    ],
)
def test_param_groups_covering_all_parameters_in_transformer_model(name):
    config = AutoConfig.from_pretrained(name)
    model = AutoModelForCustomSequenceClassification.from_config(config)

    param_groups = create_param_groups(
        model, learning_rate=1, layerwise_lr_decay=0.9, weight_decay=0.01
    )

    params = set()
    for pg in param_groups:
        params.update(list(pg["params"]))

    assert {p for n, p in model.named_parameters() if "mask_emb" not in n} == params
