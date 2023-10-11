import importlib
from popgym.wrappers import (
    Antialias,
    PreviousAction,
    Flatten,
    DiscreteAction,
)
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax

def scale_by_norm(scale: float=1.0, eps: float=1e-6):
  def init_fn(params):
    del params
    return optax._src.base.EmptyState()

  def update_fn(updates, state, params=None):
    del params
    g_norm = jnp.maximum(optax.global_norm(updates) + eps, 1 / scale)
    #g_norm = (optax.global_norm(updates) / scale + eps)
    def scale_fn(t):
       return t / g_norm

    updates = jax.tree_util.tree_map(scale_fn, updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


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
