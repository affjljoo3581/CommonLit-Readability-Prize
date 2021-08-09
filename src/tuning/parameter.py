from __future__ import annotations

import ast
import re
from typing import List, Optional, Type, Union

import numpy as np
from numpy.random import RandomState
from optuna.trial import Trial


class TunableParameter:
    """
    A tunable parameter which can be replaced with a suggestion from the experiment
    system or sampled randomly. This class supports the hierarchical experiment leveling
    to maange multi-layered experiments efficiently. It manages the parameter decision
    according to the experiment level.

    Note:
        The parameter space is decided by `low`, `high` and `choices` arguments. `low`
        and `high` are used for the range space, and `choices` is used for the
        categorical space. Therefore, you must specify `low` and `high` with
        `choices=None` or `choices` with `low=None` and `high=None`. Elsewise, a
        `ValueError` will be raised.

    Args:
        name: The parameter name. It is used to `optuna.suggest_*` method.
        level: The desired experiment level for this parameter. If the current running
            level is lower than the parameter's level, then the actual value will be
            sampled from the parameter space rather than requesting to the `optuna`.
        low: The minimum value of the parameter. Default is `None`.
        high: The maximum value of the parameter. This value is exclusive. Default is
            `None`.
        choices: The collection of the parameter values. Default is `None`.
    """

    def __init__(
        self,
        name: str,
        level: int,
        low: Union[int, float, None] = None,
        high: Union[int, float, None] = None,
        choices: Optional[List[Union[int, float]]] = None,
    ):
        self.name = name
        self.level = level
        self.low = low
        self.high = high
        self.choices = choices

        if (low is None and high is None and choices is None) or (
            low is not None and high is not None and choices is not None
        ):
            raise ValueError(
                "`low`, `high`, and `choices` cannot be either `None` or not `None`."
                " You must specify `low` and `high` with `choices=None`, or `choices`"
                " with `low=None` and `high=None`."
            )

    def suggest(self, trial: Trial) -> Union[int, float]:
        """Decide the parameter value from the given `optuna` trial.

        Args:
            trial: The current trial object to suggest the parameter value.

        Returns:
            A suggested parameter value from the given trial object.
        """
        if self.choices is not None:
            return trial.suggest_categorical(self.name, self.choices)
        elif self.param_type == int:
            return trial.suggest_int(self.name, self.low, self.high)
        else:
            return trial.suggest_float(self.name, self.low, self.high)

    def sample(self, rng: Optional[RandomState] = None) -> Union[int, float]:
        """Sample the parameter value by using the given random state.

        Args:
            rng: The optional `np.random.RandomState`. You can control the random seed
                by using this argument. If `None`, this method will use the default
                global numpy random state. Default is `None`.

        Returns:
            A sampled random parameter value.
        """
        # Use the default numpy random state if nothing is given.
        rng = rng or np.random.random.__self__

        if self.choices is not None:
            return rng.choice(self.choices).item()
        elif self.param_type == int:
            return rng.randint(self.low, self.high)
        else:
            return rng.uniform(self.low, self.high)

    def __call__(
        self, level: int, trial: Trial, rng: Optional[RandomState] = None
    ) -> Union[int, float]:
        """
        Decide the parameter value by sampling from the parameter space or suggesting
        from `optuna` system. Rather than using this method, you can explicitly perform
        sampling or suggesting by calling `sample` or `suggest` respectively. This
        method automate this decision with the given experiment level.

        Args:
            level: The current experiment level. If current level is greater than the
                desired level of this parameter, then the parameter sampling will be
                performed. If the current level equals to the desired level, then the
                parameter suggestion from `optuna` will be performed, Otherwise,
                `RuntimeError` will be raised.
            trial: The current trial object to suggest the parameter value.
            rng: The optional `np.random.RandomState`. You can control the random seed
                by using this argument. If `None`, this method will use the default
                global numpy random state. Default is `None`.

        Returns:
            A suggested or randomly sampled parameter value.
        """
        if level == self.level:
            return self.suggest(trial)
        elif level < self.level:
            return self.sample(rng)
        else:
            raise RuntimeError(
                f"the level of this parameter is {self.level}, but the current running "
                f"level is {level}. this parameter should be overrided before "
                f"suggesting or sampling."
            )

    @property
    def possible_values(self) -> List[Union[int, float]]:
        """Return all possible values for the parameter."""
        if self.choices is not None:
            return self.choices
        elif self.param_type == float:
            raise TypeError("cannot collect possible values on continuous float space")
        return list(range(self.low, self.high + 1))

    @property
    def param_type(self) -> Type:
        """Return the type of the parameter."""
        targets = self.choices or [self.low, self.high]
        return int if all(isinstance(x, int) for x in targets) else float

    @staticmethod
    def is_parameter(format: str) -> bool:
        """Check if the string is tunable-parameter format."""
        return re.match(r"\$\d+:.+", format) is not None

    @staticmethod
    def parse(name: str, format: str) -> TunableParameter:
        """Parse the tunable-parameter formatted string.

        Args:
            format: The tunable-parameter formatted string.

        Returns:
            A new `TunableParameter` object parsed from the given string.
        """
        level, domain = re.match(r"\$(\d+):(.+)", format).groups()
        domain = ast.literal_eval(domain)

        if isinstance(domain, tuple):
            return TunableParameter(name, int(level), low=domain[0], high=domain[1])
        else:
            return TunableParameter(name, int(level), choices=domain)
