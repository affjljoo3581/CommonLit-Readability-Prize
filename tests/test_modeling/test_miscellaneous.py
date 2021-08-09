import pytest
import torch
from transformers import AutoConfig

from modeling.miscellaneous import (
    AttentionBasedClassificationHead,
    SequenceClassificationLoss,
)


@pytest.mark.parametrize("shape", [(), (5,), (16, 32), (5, 4, 3)])
def test_sequence_classification_loss_with_regression(shape):
    config = AutoConfig.from_pretrained("bert-base-cased", num_labels=1)
    criterion = SequenceClassificationLoss(config)

    logits = torch.rand(shape)
    labels = torch.rand(shape)

    assert criterion(logits, labels).shape == ()
    assert config.problem_type == "regression"


@pytest.mark.parametrize("shape", [(), (5,), (16, 32), (5, 4, 3)])
@pytest.mark.parametrize("dim", [4, 8, 16])
def test_sequence_classification_loss_with_single_label_classification(shape, dim):
    config = AutoConfig.from_pretrained("bert-base-cased", num_labels=dim)
    criterion = SequenceClassificationLoss(config)

    logits = torch.rand(shape + (dim,))
    labels = torch.randint(0, dim, shape)

    assert criterion(logits, labels).shape == ()
    assert config.problem_type == "single_label_classification"


@pytest.mark.parametrize("shape", [(), (5,), (16, 32), (5, 4, 3)])
@pytest.mark.parametrize("dim", [4, 8, 16])
def test_sequence_classification_loss_with_multi_label_classification(shape, dim):
    config = AutoConfig.from_pretrained("bert-base-cased", num_labels=dim)
    criterion = SequenceClassificationLoss(config)

    logits = torch.rand(shape + (dim,))
    labels = torch.rand(shape + (dim,))

    assert criterion(logits, labels).shape == ()
    assert config.problem_type == "multi_label_classification"


@pytest.mark.parametrize("shape", [(1, 1, 128), (4, 16, 32), (16, 4, 64)])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("num_labels", [1, 4, 8, 16])
def test_attention_based_classification_head_output_with_correct_shape(
    shape, use_mask, num_labels
):
    classifier = AttentionBasedClassificationHead(shape[-1], num_labels, 0.1)

    last_hidden_state = torch.rand(shape)
    attention_mask = torch.randint(0, 1, shape[:-1]) if use_mask else None

    logits = classifier(last_hidden_state, attention_mask)
    assert logits.shape == shape[:-2] + (num_labels,)
