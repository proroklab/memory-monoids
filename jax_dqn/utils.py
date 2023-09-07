import importlib
from popgym.wrappers import (
    Antialias,
    PreviousAction,
    Flatten,
    DiscreteAction,
    EpisodeStart,
)
import gymnasium as gym
import jax


def load_popgym_env(config, eval=False, popgym=True):
    if popgym:
        module, cls = config["collect"]["env"].rsplit(".", 1)
        mod = importlib.import_module(module)
        instance = getattr(mod, cls)()
    else:
        instance = gym.make(config["collect"]["env"])
    instance = Flatten(Antialias(PreviousAction(instance)))
    if isinstance(instance.action_space, gym.spaces.MultiDiscrete):
        instance = DiscreteAction(instance)
    instance.action_space.seed(config["seed"] + eval * 1000)

    return instance


def filter_inf(log_dict):
    d = {}
    for k, v in log_dict.items():
        if k != float("-inf"):
            d[k] = v
    return d


@jax.jit
def expand_right(src, shape):
    a_dims = len(src.shape)
    b_dims = len(shape)
    right = [1] * (a_dims - b_dims)
    return src.reshape(*src.shape, *right)
