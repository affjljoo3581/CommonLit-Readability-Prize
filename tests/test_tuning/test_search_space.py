from functools import partial

import optuna
import pytest
from omegaconf import OmegaConf
from optuna.trial import FixedTrial

from tuning.search_space import SearchSpace


@pytest.mark.parametrize("use_omegaconf", [True, False])
def test_search_space_collecting_parameters(use_omegaconf):
    config = {
        "model": {"encoder": {"num_layers": 12}, "decoder": {"num_layers": 6}},
        "optim": {
            "learning_rate": "$0:(1e-5, 1e-4)",
            "weight_decay": "$0:[0, 0.01, 0.1]",
        },
        "train": {"epochs": "$1:(3, 6)"},
    }
    if use_omegaconf:
        config = OmegaConf.create(config)

    search_space = SearchSpace(config)
    assert set(search_space.parameters.keys()) == {
        "optim.learning_rate",
        "optim.weight_decay",
        "train.epochs",
    }


@pytest.mark.parametrize("use_omegaconf", [True, False])
def test_search_space_collecting_possible_values(use_omegaconf):
    config = {
        "model": {"encoder": {"num_layers": 12}, "decoder": {"num_layers": 6}},
        "optim": {
            "learning_rate": "$0:(1e-5, 1e-4)",
            "weight_decay": "$0:[0, 0.01, 0.1]",
        },
        "train": {"epochs": "$1:(3, 6)"},
    }
    if use_omegaconf:
        config = OmegaConf.create(config)

    search_space = SearchSpace(config)

    search_space.set_experiment_level(0)
    with pytest.raises(TypeError):
        search_space.possible_values

    search_space.set_experiment_level(1)
    assert search_space.possible_values == {"train.epochs": [3, 4, 5, 6]}


@pytest.mark.parametrize("use_omegaconf", [True, False])
def test_search_space_sampling_invariant(use_omegaconf):
    trial = FixedTrial(
        {"optim.learning_rate": 5e-5, "optim.weight_decay": 0.01, "train.epochs": 5}
    )

    config = {
        "model": {"encoder": {"num_layers": 12}, "decoder": {"num_layers": 6}},
        "optim": {
            "learning_rate": "$2:(1e-5, 1e-4)",
            "weight_decay": "$0:[0, 0.01, 0.1]",
        },
        "train": {"epochs": "$1:(3, 6)"},
    }
    if use_omegaconf:
        config = OmegaConf.create(config)

    search_space = SearchSpace(config)

    search_space.set_experiment_level(0)
    config0 = search_space(trial)

    search_space.set_experiment_level(1)
    search_space.update_parameters({"optim.weight_decay": 0})
    config1 = search_space(trial)

    assert config0["optim"]["learning_rate"] == config1["optim"]["learning_rate"]


@pytest.mark.parametrize("use_omegaconf", [True, False])
def test_search_space_optuna_optimization(use_omegaconf):
    config = {
        "model": {"encoder": {"num_layers": 12}, "decoder": {"num_layers": 6}},
        "optim": {
            "learning_rate": "$0:(1e-5, 1e-4)",
            "weight_decay": "$2:[0, 0.01, 0.1]",
        },
        "train": {"epochs": "$1:(3, 6)"},
    }
    if use_omegaconf:
        config = OmegaConf.create(config)

    def objective(trial, search_space):
        config = search_space(trial)

        assert 1e-5 <= config["optim"]["learning_rate"] < 1e-4
        assert config["optim"]["weight_decay"] in [0, 0.01, 0.1]
        assert 3 <= config["train"]["epochs"] <= 6

        return 0

    search_space = SearchSpace(config)
    for i in range(3):
        search_space.set_experiment_level(i)

        study = optuna.create_study()
        study.optimize(partial(objective, search_space=search_space), n_trials=25)

        search_space.update_parameters(study.best_params)
