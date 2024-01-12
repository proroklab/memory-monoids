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
import numpy as np

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
        instance = getattr(mod, cls)(**config["collect"].get("env_kwargs", {}))
    else:
        instance = gym.make(config["collect"]["env"])
    if config["collect"]["env_prev_action"]:
        instance = PreviousAction(instance)
    instance = Flatten(Antialias(instance))
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

def get_summary_info(model):
    """An alternative repr useful for initial debugging"""
    import pandas as pd

    def get_info(v):
        info = dict()
        info['type'] = type(v).__name__
        info['dtype'] = v.dtype.name if hasattr(v, 'dtype') else None
        info['shape'] = jnp.shape(v)
        info['size'] = jnp.size(v)
        #info['nancount'] = np.isnan(v).sum()
        #info['zerocount'] = np.size(v) - np.count_nonzero(v)
        info['min'] = jnp.min(v).item()
        info['max'] = jnp.max(v).item()
        info['mean'] = jnp.mean(v).item()
        info['std'] = jnp.std(v).item()
        info['norm'] = jnp.linalg.norm(v).item()
        return info

    d_ = {jax.tree_util.keystr(k): get_info(v) for k, v in jax.tree_util.tree_leaves_with_path(model) if isinstance(v, (jax.Array, float))}
    return pd.DataFrame(d_).T

def get_wandb_model_info(model):
    """An alternative repr useful for initial debugging"""
    info = {}
    for k, v in jax.tree_util.tree_leaves_with_path(model):
        if isinstance(v, (jax.Array)):
            prefix = "params/model"
            k = jax.tree_util.keystr(k)
            info[prefix + k + '.mean'] = jnp.mean(v)
            info[prefix + k + '.std'] = jnp.std(v)
            info[prefix + k + '.norm'] = jnp.linalg.norm(v)
    return info


def elementwise_grad(g):
  def wrapped(x, *rest):
    y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
    x_bar, = g_vjp(jnp.ones_like(y))
    return x_bar
  return wrapped