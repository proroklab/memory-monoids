from typing import Any, Dict
import jax
import equinox as eqx
from equinox import nn
from jax import random, vmap
import jax.numpy as jnp
from modules import mish, FinalLinear


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
        self.pre = nn.Sequential(
            [nn.Linear(obs_shape, config["mlp_size"], key=keys[0]), mish]
        )
        self.memory = nn.GRUCell(
            config["mlp_size"], self.config["recurrent_size"], key=keys[1]
        )
        self.post = nn.Sequential(
            [
                nn.Linear(
                    self.config["recurrent_size"], self.config["mlp_size"], key=keys[2]
                ),
                mish,
                nn.Linear(
                    self.config["mlp_size"], self.config["mlp_size"], key=keys[3]
                ),
                mish,
                # nn.Dropout(p=0.01),
                FinalLinear(self.config["mlp_size"], self.output_size, key=keys[4]),
            ]
        )

    @eqx.filter_jit
    def __call__(self, x, state, start, done):
        x = vmap(self.pre)(x)

        def scan_fn(state, input):
            x, start, done = input
            state = self.memory(x, state)
            return state, state

        final_state, state = jax.lax.scan(scan_fn, state, (x, start, done))
        y = vmap(self.post)(state)
        return y, final_state

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, self.config["recurrent_size"]), dtype=jnp.float32)
