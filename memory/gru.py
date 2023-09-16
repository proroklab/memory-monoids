from typing import Any, Dict, List
import jax
import equinox as eqx
from equinox import nn
from jax import random, vmap
import jax.numpy as jnp
from modules import mish, ortho_linear, final_linear


class GRU(eqx.Module):
    input_size: int
    recurrent_size: int
    gru: eqx.Module
    name: str = "GRU"

    def __init__(self, input_size, recurrent_size, key):
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        _, key = random.split(key)
        self.gru= nn.GRUCell(
            input_size, recurrent_size, key=key
        ) 

    @eqx.filter_jit
    def scan_fn(self, state, input):
        x, start = input
        state = self.gru(x, state * jnp.logical_not(start))
        return state, state
    
    @eqx.filter_jit
    def __call__(self, x, state, start, next_done, key):
        final_state, y = jax.lax.scan(self.scan_fn, state, (x, start))
        return y, final_state

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, self.recurrent_size), dtype=jnp.float32)