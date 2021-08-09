import argparse
from typing import Any, Iterator, Mapping, Tuple


def iterate_hierarchical_dict(data: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Iterate a hierarchical dictionary and yield items with an integrated names. This
    function uses internal reducing function and calls it recursively to resolve the
    hierarichal dictionary structure. This class helps to iterate the complex
    multi-layered dictionary simply by handling it as a single dictionary.

    Args:
        data: The multi-layered complex mapping object. It can be either native python
            dictionary or `omegaconf.DictConfig`.

    Yields:
        The tuple of the integrated key and its value.
    """

    def _reduce_dict_level(data: Any) -> Iterator[Tuple[str, Any]]:
        if isinstance(data, Mapping):
            for key, value in data.items():
                # Recursively call this function and collect items with merging the
                # hierarchical prefix.
                for param_key, param_value in _reduce_dict_level(value):
                    param_key = f"{key}.{param_key}" if param_key else key
                    yield param_key, param_value
        else:
            yield "", data

    yield from _reduce_dict_level(data)


def override_from_dict(config: Mapping[str, Any], overrides: Mapping[str, Any]):
    """
    Override the configuration parameters from the mapping object. Note that the
    configuration should be hierarchical but overriding dictionary can have either
    hierarchical or integrated names. See the example below:

    Examples::

        >>> config = {"A": {"B": 1}, "C": {"D": 2, "E": 3}}
        >>> overrides = {"A": {"B": 4}, "C.D": 5, "C": {"E": 6}}
        >>> override_from_dict(config, overrides)
        >>> config
        {"A": {"B": 4}, "C": {"D": 5, "E": 6}}

    Args:
        config: The target configuration mapping object. It can be either native
            dictionary or `omegaconf.DictConfig`.
        overrides: The overriding mapping object. It can be either native dictionary or
            `omegaconf.DictConfig`.
    """
    for key, value in overrides.items():
        target = config
        names = key.split(".")

        # Access to the last container.
        for name in names[:-1]:
            target = target[name]

        # If the overriding value is a dictionary, then call this function recursively
        # to modify the values in the dictionary.
        if isinstance(value, Mapping):
            override_from_dict(target[names[-1]], value)
        else:
            target[names[-1]] = value


def override_from_argparse(config: Mapping[str, Any]):
    """
    Override the configuration paramters from argparse command-line inputs. Using this
    function, you can modify the configuration parameters by using command-line
    arguments. The argument names must be the integrated (or reduced) parameter names in
    the givne configuration.

    Examples:
        First of all, write the script to read a configuration file and override from
        cli::

            >>> parser = argparse.ArgumentParser()
            >>> parser.add_argument("config")
            >>> args = parser.parse_args()
            >>> config = OmegaConf.load(args.config)
            >>> override_from_argparse(config)
            >>> print(config)

        And run the script with parameter modifications::

            $ python script.py config.yaml
            {"optimizer": {"learning_rate": 1e-4, "weight_decay": 0.01}}
            $ python script.py config.yaml --optimizer.learning_rate 1e-5
            {"optimizer": {"learning_rate": 1e-5, "weight_decay": 0.01}}

    Args:
        config: The target configuration mapping object. It can be either native
            dictionary or `omegaconf.DictConfig`.
    """
    from tuning import TunableParameter

    # Add optional arguments to the parser.
    parser = argparse.ArgumentParser()

    for key, value in iterate_hierarchical_dict(config):
        if isinstance(value, str) and TunableParameter.is_parameter(value):
            param = TunableParameter.parse(key, value)
            parser.add_argument(f"--{key}", type=param.param_type)
        else:
            parser.add_argument(f"--{key}", type=type(value))

    # Parse the command-line arguments and override to the configuration. Note that the
    # `None` parameters (which are not given from cli) will be excluded.
    args, _ = parser.parse_known_args()
    override_from_dict(config, {k: v for k, v in vars(args).items() if v is not None})
