import copy
from typing import Any, Mapping, Optional, Tuple, Union

import optuna
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
)
from optuna.samplers import BaseSampler, GridSampler, RandomSampler, TPESampler
from optuna.study import Study

from tuning.search_space import SearchSpace


def _create_sampler(
    config: Union[str, Mapping[str, Any], None], search_space: SearchSpace
) -> Optional[BaseSampler]:
    # Normalize the configuration structure.
    if config is None:
        return None
    elif isinstance(config, str):
        config = {"type": config}
    elif "type" not in config:
        raise TypeError("class type must be specified.")

    # Copy the configuration and pop `type` key to preserve the original one.
    config = copy.deepcopy(config)
    class_type = config.pop("type")

    # Create new sampler with keyword-arguments.
    if class_type == "grid":
        return GridSampler(search_space.possible_values, **config)
    elif class_type == "random":
        return RandomSampler(**config)
    elif class_type == "tpe":
        return TPESampler(**config)
    else:
        raise TypeError(f"sampler {class_type} is not supported.")


def _create_pruner(
    config: Union[str, Mapping[str, Any], None], search_space: SearchSpace
) -> Optional[BasePruner]:
    # Normalize the configuration structure.
    if config is None:
        return NopPruner()
    elif isinstance(config, str):
        config = {"type": config}
    elif "type" not in config:
        raise TypeError("class type must be specified.")

    # Copy the configuration and pop `type` key to preserve the original one.
    config = copy.deepcopy(config)
    class_type = config.pop("type")

    # Create new pruner with keyword-arguments.
    if class_type == "nop":
        return NopPruner()
    elif class_type == "hyperband":
        return HyperbandPruner(**config)
    elif class_type == "median":
        return MedianPruner(**config)
    elif class_type == "percentile":
        return PercentilePruner(**config)
    elif class_type == "sha":
        return SuccessiveHalvingPruner(**config)
    else:
        raise TypeError(f"pruner {class_type} is not supported.")


def create_study(
    config: Mapping[str, Any], search_space: SearchSpace
) -> Tuple[Study, Optional[int]]:
    """Create new study with simplified configuration.

    This function creates new `optuna.study.Study` object automatically, without complex
    detailed settings. Almost every optimization settings can be defined by using
    simplified configuration structure. You can specify your own sampler, pruner and the
    number of trials. They can be also `None` when they are not specified.

    Examples:
        First of all, create new YAML config file like below:
        .. code-block:: yaml

            experiments:
              - method: grid
                early_stop: median
              - method:
                  type: tpe
                  multivariate: true
                  group: true
                early_stop: median
                num_trials: 100

            config:
                ...

        After writing your own config file, you can execute the experiments as follows:
        .. code-block:: python

            search_space = SearchSpace(config.config)
            for i, experiment in enumerate(config.experiments):
                search_space.set_experiment_level(i)

                study, n_trials = create_study(experiment, search_space)
                study.optimize(partial(objective, search_space=search_space), n_trials)

                search_space.update_parameters(study.best_params)

    Args:
        config: The configuration mapping object containing informations about sampler,
            pruner and a number of trials for optimization.
        search_space: The `SearchSpace` object. It is used to collect the grid-sampling
            space when creating a grid sampler.

    Returns:
        - A new study with corresponding sampler and pruner.
        - The number of trials specified in the configuration.
    """
    study = optuna.create_study(
        sampler=_create_sampler(config.get("method", None), search_space),
        pruner=_create_pruner(config.get("early_stop", None), search_space),
    )
    return study, config.get("num_trials", None)
