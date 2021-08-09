from omegaconf import OmegaConf
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import GridSampler, TPESampler

from tuning.search_space import SearchSpace
from tuning.study import create_study


def test_creating_study_type_a():
    config = OmegaConf.create(
        """
        experiment:
          method: grid

        config:
          param1: $0:(0, 100)
        """
    )

    search_space = SearchSpace(config.config)
    study, n_trials = create_study(config.experiment, search_space)

    assert isinstance(study.sampler, GridSampler)
    assert isinstance(study.pruner, NopPruner)
    assert n_trials is None


def test_creating_study_type_b():
    config = OmegaConf.create(
        """
        experiment:
          early_stop: hyperband
          num_trials: 100

        config:
          param1: $0:(0, 100)
        """
    )

    search_space = SearchSpace(config.config)
    study, n_trials = create_study(config.experiment, search_space)

    assert isinstance(study.sampler, TPESampler)
    assert isinstance(study.pruner, HyperbandPruner)
    assert n_trials == 100


def test_creating_study_type_c():
    config = OmegaConf.create(
        """
        experiment:
          method:
            type: tpe
            multivariate: true
            group: true
          early_stop:
            type: hyperband
            min_resource: 10
          num_trials: 10

        config:
          param1: $0:(0, 100)
        """
    )

    search_space = SearchSpace(config.config)
    study, n_trials = create_study(config.experiment, search_space)

    assert isinstance(study.sampler, TPESampler)
    assert isinstance(study.pruner, HyperbandPruner)
    assert study.sampler._multivariate
    assert study.sampler._group
    assert study.pruner._min_resource == 10
    assert n_trials == 10
