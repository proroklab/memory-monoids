from typing import Any, Dict
import jax
import equinox as eqx
from equinox import nn
from jax import random, vmap
import jax.numpy as jnp
from modules import mish, FinalLinear, final_layer_init, NoisyLinear


class StochasticSequential(nn.Sequential):
    def __call__(self, x, key):
        return super().__call__(x, key=key)


class GRUQNetwork(eqx.Module):
    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: eqx.Module
    post: eqx.Module
    name: str = "GRU"

    def __init__(self, obs_shape, act_shape, config, key):
        self.config = config
        self.input_size = obs_shape
        self.output_size = act_shape
        keys = random.split(key, 5)
        pre = nn.Sequential(
            [nn.Linear(obs_shape, config["mlp_size"], key=keys[0]), mish]
        )
        self.pre = eqx.filter_vmap(pre)
        self.memory = nn.GRUCell(
            config["mlp_size"], self.config["recurrent_size"], key=keys[1]
        )
        post = nn.Sequential(
            [
                nn.Linear(
                    self.config["recurrent_size"], self.config["mlp_size"], key=keys[2]
                ),
                mish,
                nn.Linear(
                    self.config["mlp_size"], self.config["mlp_size"], key=keys[3]
                ),
                mish,
                final_layer_init(nn.Linear(self.config["mlp_size"], self.output_size, key=keys[4])),
            ]
        )
        self.post = eqx.filter_vmap(post)

    @eqx.filter_jit
    def scan_fn(self, state, input):
        x, start, done = input
        state = self.memory(x, state * jnp.logical_not(start))
        return state, state

    @eqx.filter_jit
    def __call__(self, x, state, start, done, key):
        #key, pre_key, post_key = random.split(key, 3)
        x = self.pre(x)
        final_state, state = jax.lax.scan(self.scan_fn, state, (x, start, done))
        y = self.post(state)
        return y, final_state

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, self.config["recurrent_size"]), dtype=jnp.float32)
