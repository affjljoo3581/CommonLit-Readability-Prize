import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from modeling.models.xlnet import XLNetForCustomSequenceClassification


@pytest.mark.parametrize("num_labels", [1, 4])
@pytest.mark.parametrize("return_dict", [True, False])
def test_xlnet_for_custom_sequence_classification_forwarding(num_labels, return_dict):
    config = AutoConfig.from_pretrained("xlnet-base-cased", num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

    encodings = tokenizer(
        ["Hello World!"] * 2,
        padding="max_length",
        max_length=8,
        return_tensors="pt",
    )
    labels = torch.randint(0, num_labels, (2,))

    model = XLNetForCustomSequenceClassification(config)
    outputs = model(labels=labels, return_dict=return_dict, **encodings)

    if return_dict:
        assert outputs[0].shape == ()
        assert outputs[1].shape == (2, num_labels)
    else:
        assert outputs.loss is not None
        assert outputs.loss.shape == ()
        assert outputs.logits.shape == (2, num_labels)


@pytest.mark.parametrize("dropout_prob", [0.0, 0.1, 0.2, 0.5])
def test_xlnet_for_custom_sequence_classification_modifying_dropout(dropout_prob):
    config = AutoConfig.from_pretrained("xlnet-base-cased")
    model = XLNetForCustomSequenceClassification(config)

    model.set_classifier_dropout(dropout_prob)
    for module in model.classifier.modules():
        if isinstance(module, nn.Dropout):
            assert module.p == dropout_prob


def test_xlnet_for_custom_sequence_classification_initializing_classifier():
    initialized_modules = set()

    def _modified_init_weights(module):
        initialized_modules.add(module)

    config = AutoConfig.from_pretrained("xlnet-base-cased")
    model = XLNetForCustomSequenceClassification(config)
    model._init_weights = _modified_init_weights

    model.init_classifier()
    assert set(model.classifier.modules()) == initialized_modules


@pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
@pytest.mark.parametrize("reverse", [True, False])
def test_xlnet_for_custom_sequence_classification_initializing_transformer_layers(
    num_layers, reverse
):
    initialized_modules = set()

    def _modified_init_weights(module):
        initialized_modules.add(module)

    config = AutoConfig.from_pretrained("xlnet-base-cased")
    model = XLNetForCustomSequenceClassification(config)
    model._init_weights = _modified_init_weights

    model.init_transformer_layers(num_layers, reverse)

    layers = list(model.hierarchical_modules())[1:-1]
    layers = layers[: -num_layers - 1 : -1] if reverse else layers[:num_layers]
    assert initialized_modules == set(sum([list(m) for m in layers], []))


def test_xlnet_for_custom_sequence_classification_covering_all_parameters():
    config = AutoConfig.from_pretrained("xlnet-base-cased")
    model = XLNetForCustomSequenceClassification(config)

    params = set()
    for modules in model.hierarchical_modules():
        for module in modules:
            params.update(module.parameters(recurse=False))

    assert {p for n, p in model.named_parameters() if "mask_emb" not in n} == params
