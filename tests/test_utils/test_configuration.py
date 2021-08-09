import sys

import pytest
from omegaconf import OmegaConf

from utils.configuration import (
    iterate_hierarchical_dict,
    override_from_argparse,
    override_from_dict,
)


@pytest.mark.parametrize("use_omegaconf", [True, False])
def test_iterating_hierarchical_dict(use_omegaconf):
    config = {
        "model": {"encoder": {"num_layers": 12}, "decoder": {"num_layers": 6}},
        "optim": {"learning_rate": 1e-5, "weight_decay": 0.01},
        "train": {"epochs": 10},
    }
    if use_omegaconf:
        config = OmegaConf.create(config)

    assert set(iterate_hierarchical_dict(config)) == {
        ("model.encoder.num_layers", 12),
        ("model.decoder.num_layers", 6),
        ("optim.learning_rate", 1e-5),
        ("optim.weight_decay", 0.01),
        ("train.epochs", 10),
    }


@pytest.mark.parametrize("use_omegaconf_config", [True, False])
@pytest.mark.parametrize("use_omegaconf_overrides", [True, False])
def test_overriding_from_dict(use_omegaconf_config, use_omegaconf_overrides):
    config = {
        "model": {"encoder": {"num_layers": 12}, "decoder": {"num_layers": 6}},
        "optim": {"learning_rate": 1e-5, "weight_decay": 0.01},
        "train": {"epochs": 10},
    }
    overrides = {
        "model": {"encoder.num_layers": 6},
        "model.decoder": {"num_layers": 3},
        "train.epochs": 3,
    }

    if use_omegaconf_config:
        config = OmegaConf.create(config)
    if use_omegaconf_overrides:
        overrides = OmegaConf.create(overrides)

    override_from_dict(config, overrides)
    assert set(iterate_hierarchical_dict(config)) == {
        ("model.encoder.num_layers", 6),
        ("model.decoder.num_layers", 3),
        ("optim.learning_rate", 1e-5),
        ("optim.weight_decay", 0.01),
        ("train.epochs", 3),
    }


@pytest.mark.parametrize("use_omegaconf", [True, False])
def test_overriding_from_argparse(use_omegaconf):
    config = {
        "model": {"encoder": {"num_layers": 12}, "decoder": {"num_layers": 6}},
        "optim": {
            "learning_rate": "$0:(1e-5, 1e-4)",
            "weight_decay": "$1:[0, 0.01, 0.1]",
        },
        "train": {"epochs": "$2:(3, 6)"},
    }
    if use_omegaconf:
        config = OmegaConf.create(config)

    sys.argv = (
        "python train.py --model.decoder.num_layers 12"
        "                --optim.learning_rate 5e-5"
        "                --train.epochs 4"
    ).split()

    override_from_argparse(config)
    assert set(iterate_hierarchical_dict(config)) == {
        ("model.encoder.num_layers", 12),
        ("model.decoder.num_layers", 12),
        ("optim.learning_rate", 5e-5),
        ("optim.weight_decay", "$1:[0, 0.01, 0.1]"),
        ("train.epochs", 4),
    }
