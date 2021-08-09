import copy
from typing import Any, Dict, List, Mapping, Union

import numpy as np
from optuna.trial import Trial

from tuning.parameter import TunableParameter
from utils import iterate_hierarchical_dict, override_from_dict


class SearchSpace:
    """
    A fully-automated search-space manager which controls a configuration with tunable
    parameters. This class helps to manage the configuration of each experiment simply
    by separating the tunable parameters by their experiment levels. This class supports
    hierarchical parameter suggestion with multi-layered experiments and overriding
    parameters from previous experiment results.

    It internally separates the tunable parameters from the configuration (fixed
    parameter set). When a new configuration with suggestions is required, it modifies
    the base configuration by calling `optuna.suggest_*` or sampling randomly and
    repacks to the original class. Therefore, if you are using a native python
    dictionary then new configuration object will also be python dictionary, and if
    using `omegaconf.DictConfig` then it will be `omegaconf.DictConfig` as well.

    If the level of the tunable parameter is higher than the current experiment level,
    then the value will be decided by sampling from the target domain space. Since the
    parameters will be overrided sequentially, the random sampling can be changed. Hence
    this class samples the higher-level parameters reversely, so that the random
    sampling does not changed.

    Examples::

        def objective(trial, search_space):
            config = search_space(trial)
            ...

        search_space = SearchSpace(config)
        for i in range(...):
            search_space.set_experiment_level(i)

            study = optuna.create_study()
            study.optimize(partial(objective, search_space=search_space), n_trials=...)

            search_space.update_parameters(study.best_params)

    Args:
        config: The default (base) configuration mapping object with formatted tunable
            parameters. This class automatically detects the tunable parameters and
            collects them.
        random_seed: The random seed for sampling the parameters of which level is
            higher than current experiment level. Using this random seed, the randomly
            sampled parameter values would be always same. Default is `0`.
    """

    def __init__(self, config: Mapping[str, Any], random_seed: int = 0):
        self.config = config
        self.random_seed = random_seed

        self.experiment_level = 0
        self.parameters = {
            key: TunableParameter.parse(key, value)
            for key, value in iterate_hierarchical_dict(config)
            if isinstance(value, str) and TunableParameter.is_parameter(value)
        }

    def set_experiment_level(self, level: int):
        """Set current experiment level.

        This class stores this value and uses it when the parameter suggestion is
        requested. This method is considered to separate the experiment-level dependency
        from the objective function. By configuring the experiment level before running
        experiments, the objective function does not need to know the current level.

        Args:
            level: The current experiment level.
        """
        self.experiment_level = level

    def update_parameters(self, new_params: Mapping[str, Any]):
        """Update the parameter values.

        Using the given parameter values, the default configuration will be overrided
        and the corresponding tunable parameters will be removed from the
        already-collected tunable-parameter list. This method is usually used for
        updating to the optimized parameter values.

        Note that the internal implementation uses
        `utils.configuration.override_from_dict`, which requires an overriding
        dictionary with either hierarchical or integrated names.

        Args:
            new_params: The overriding map object which contains new parameter values.
        """
        override_from_dict(self.config, new_params)

        # Remove the corresponding tunable parameters.
        for key in new_params:
            self.parameters.pop(key)

    @property
    def possible_values(self) -> Dict[str, List[Union[int, float]]]:
        """Collect possible values of current experiment's parameters."""
        return {
            k: v.possible_values
            for k, v in self.parameters.items()
            if v.level == self.experiment_level
        }

    def __call__(self, trial: Trial) -> Mapping[str, Any]:
        """Create new configuration based on the default one.

        This method first decides the tunable parameters, which are collected from the
        default configuration, by using `TunableParameter` class. After that, the
        decided tunable parameter values will be assigned to the new configuration
        dictionary, which is copied from the default one. During this procedures, the
        fixed random state will be used to ensure the reproducibility of each
        parameters.

        Args:
            trial: The current trial object to suggest the parameter value.

        Returns:
            A new configuration mapping object with fixed tunable parameters.
        """
        new_config = copy.deepcopy(self.config)
        rng = np.random.RandomState(self.random_seed)

        parameters = sorted(self.parameters.items(), key=lambda item: -item[1].level)
        overrides = {
            name: param(self.experiment_level, trial, rng) for name, param in parameters
        }

        override_from_dict(new_config, overrides)
        return new_config
