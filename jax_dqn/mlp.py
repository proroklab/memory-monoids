import math
from typing import Any, Dict
import jax
import equinox as eqx
from equinox import nn
from jax import random, vmap
import jax.numpy as jnp
from modules import mish, FinalLinear, NoisyLinear, final_layer_init


class MLPQNetwork(eqx.Module):
    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    #post: eqx.Module
    value: eqx.Module
    advantage: eqx.Module
    name: str = "MLP"

    def __init__(self, obs_shape, act_shape, config, key):
        self.config = config
        self.input_size = obs_shape
        self.output_size = act_shape
        keys = random.split(key, 6)
        self.pre = nn.Sequential(
            [nn.Linear(obs_shape, config["mlp_size"], key=keys[1]), mish]
        )
        self.value = nn.Sequential(
            [
                nn.Linear(
                    self.config["mlp_size"], self.config["mlp_size"], key=keys[2]
                ),
                mish,
                final_layer_init(nn.Linear(
                    self.config["mlp_size"], 1, key=keys[3]
                )
                )
            ]
        )
        self.advantage = nn.Sequential(
            [
                nn.Linear(
                    self.config["mlp_size"], self.config["mlp_size"], key=keys[4]
                ),
                mish,
                final_layer_init(nn.Linear(
                    self.config["mlp_size"], self.output_size, key=keys[5]
                )
                )
            ]
        )

    @eqx.filter_jit
    def __call__(self, x, state, start, done, key):
        x = vmap(self.pre)(x)

        value = vmap(self.value)(x)
        A = vmap(self.advantage)(x)
        advantage = A - jnp.mean(A, axis=-1, keepdims=True)
        q = value + advantage
        return q, jnp.zeros(1)

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1))
