import importlib
from popgym.wrappers import Antialias, PreviousAction, Flatten, DiscreteAction
import gymnasium as gym


def load_popgym_env(config, eval=False):
    module, cls = config['collect']["env"].rsplit(".", 1)
    mod = importlib.import_module(module)
    instance = getattr(mod, cls)()
    instance = Flatten(Antialias(PreviousAction(instance)))
    if isinstance(instance.action_space, gym.spaces.MultiDiscrete):
        instance = DiscreteAction(instance)
    instance.action_space.seed(config["seed"] + eval * 1000)

    return instance

def filter_inf(log_dict):
    d = {}
    for k, v in log_dict.items():
        if k != float('-inf'):
            d[k] = v
    return d