import numpy as np
import pytest
from optuna.trial import FixedTrial

from tuning.parameter import TunableParameter


def test_tunable_parameter_checking_parameter():
    assert TunableParameter.is_parameter("$0:(0, 10)")
    assert TunableParameter.is_parameter("$1:[1, 2, 3]")
    assert TunableParameter.is_parameter("$26:(0, 3.2)")
    assert not TunableParameter.is_parameter("0.5")
    assert not TunableParameter.is_parameter("$2:")


def test_tunable_parameter_parsing():
    param = TunableParameter.parse("name", "$0:(0, 3)")
    assert param.name == "name"
    assert param.level == 0
    assert param.low == 0
    assert param.high == 3
    assert param.choices is None

    param = TunableParameter.parse("name", "$54:[10, 11, 12]")
    assert param.name == "name"
    assert param.level == 54
    assert param.low is None
    assert param.high is None
    assert param.choices == [10, 11, 12]


def test_tunable_parameter_invalid_construction():
    with pytest.raises(ValueError):
        TunableParameter("", 0, low=None, high=None, choices=None)
    with pytest.raises(ValueError):
        TunableParameter("", 0, low=0, high=10, choices=[0, 1, 2])


def test_tunable_parameter_possible_values():
    assert TunableParameter.parse("", "$0:(0, 9)").possible_values == list(range(0, 10))
    assert TunableParameter.parse("", "$10:[1, 2, 3]").possible_values == [1, 2, 3]

    with pytest.raises(TypeError):
        TunableParameter.parse("", "$0:(0, 1.0)").possible_values


def test_tunable_parameter_type_inferencing():
    assert TunableParameter.parse("", "$0:(0, 9)").param_type == int
    assert TunableParameter.parse("", "$1:(0, 1.0)").param_type == float
    assert TunableParameter.parse("", "$32:(0.0, 1)").param_type == float
    assert TunableParameter.parse("", "$1:[1, 2, 3]").param_type == int
    assert TunableParameter.parse("", "$15:[1, 2, 3.0]").param_type == float


def test_tunable_parameter_suggesting():
    trial = FixedTrial({"a": 0, "b": 1.0, "c": 32})

    param1 = TunableParameter.parse("a", "$0:(0, 5)")
    param2 = TunableParameter.parse("b", "$2:(0, 2.0)")
    param3 = TunableParameter.parse("c", "$1:[16, 32, 64]")

    assert 0 <= param1.suggest(trial) <= 5
    assert 0 <= param2.suggest(trial) < 2.0
    assert param3.suggest(trial) in [16, 32, 64]


@pytest.mark.parametrize("use_random_state", [True, False])
def test_tunable_parameter_sampling(use_random_state):
    rng = np.random.RandomState(0) if use_random_state else None

    param1 = TunableParameter.parse("a", "$0:(0, 5)")
    param2 = TunableParameter.parse("b", "$2:(0, 2.0)")
    param3 = TunableParameter.parse("c", "$1:[16, 32, 64]")

    assert 0 <= param1.sample(rng) <= 5
    assert 0 <= param2.sample(rng) < 2.0
    assert param3.sample(rng) in [16, 32, 64]


@pytest.mark.parametrize("use_random_state", [True, False])
def test_tunable_parameter_integrated_decision(use_random_state):
    trial = FixedTrial({"a": 0, "b": 1.0, "c": 32})
    rng = np.random.RandomState(0) if use_random_state else None

    param1 = TunableParameter.parse("a", "$0:(0, 5)")
    param2 = TunableParameter.parse("b", "$1:(0, 2.0)")
    param3 = TunableParameter.parse("c", "$2:[16, 32, 64]")

    # Do experiment-0 decisions.
    param1_level0 = param1(0, trial, rng)
    param2_level0 = param2(0, trial, rng)
    param3_level0 = param3(0, trial, rng)

    assert 0 <= param1_level0 <= 5
    assert 0 <= param2_level0 < 2.0
    assert param3_level0 in [16, 32, 64]

    # Do experiment-1 decisions.
    with pytest.raises(RuntimeError):
        param1(1, trial, rng)
    param2_level1 = param2(1, trial, rng)
    param3_level1 = param3(1, trial, rng)

    assert 0 <= param2_level1 < 2.0
    assert param3_level1 in [16, 32, 64]

    # Do experiment-2 decisions.
    with pytest.raises(RuntimeError):
        param1(2, trial, rng)
    with pytest.raises(RuntimeError):
        param2(2, trial, rng)
    param3_level2 = param3(1, trial, rng)

    assert param3_level2 in [16, 32, 64]
