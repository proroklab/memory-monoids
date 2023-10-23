from typing import Any, Callable, Dict, List
import equinox as eqx
from equinox import nn
import jax
from jax import random, vmap, lax
import jax.numpy as jnp
from modules import Lambda
from modules import mish
from utils import expand_right


def phi(x, key=None):
    return 1 + jax.nn.elu(x)

class LinearAttention(eqx.Module):
    key: eqx.Module
    value: eqx.Module
    query: eqx.Module
    mlp: eqx.Module
    ln: eqx.Module
    skip: eqx.Module
    key_size: int
    value_size: int
    name: str = "LinearAttention"

    def __init__(self, input_size, key_size, value_size, key):
        keys = random.split(key, 7)
        self.value_size = value_size
        self.key_size = key_size
        hidden_size = key_size * value_size
        self.key = eqx.filter_vmap(nn.Sequential(
            [
                nn.Linear(input_size, key_size, use_bias=False, key=keys[0]),
                phi
            ]
        ))
        self.query = eqx.filter_vmap(nn.Sequential(
            [
                nn.Linear(input_size, key_size, use_bias=False, key=keys[1]),
                phi
            ]
        ))
        self.value = eqx.filter_vmap(nn.Linear(input_size, value_size, key=keys[2]))
        self.skip = eqx.filter_vmap(nn.Linear(input_size, hidden_size, key=keys[3]))
        self.mlp = eqx.filter_vmap(nn.Sequential(
            [
                nn.Linear(value_size, hidden_size, key=keys[4]),
                mish,
                nn.Linear(hidden_size, hidden_size, key=keys[5]),
                mish,
                nn.Linear(hidden_size, hidden_size, key=keys[6]),
            ]
        ))

        self.ln = eqx.filter_vmap(nn.LayerNorm(hidden_size))

    def associative_update(self, carry, inputs):
        s, z, prev_start, done = carry
        kv, k, start, next_done = inputs

        s = s * jnp.logical_not(start) + kv
        z = z * jnp.logical_not(start) + k

        return (s, z, jnp.logical_or(start, prev_start), next_done)

    def __call__(self, x, state, start, next_done, key=None):
        s, z = state
        T = x.shape[0]

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        kv = jnp.einsum("ti, tj -> tij", key, value)

        # Insert previous recurrent state
        start = start.reshape(T, 1, 1)
        next_done = next_done.reshape(T, 1, 1)
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)
        next_done = jnp.concatenate([jnp.zeros_like(next_done[:1]), next_done], axis=0)

        # T + 1 keys/values, discord the zeroth before returning
        kv = jnp.concatenate([s, kv], axis=0)
        key = jnp.concatenate([z, key.reshape(T, 1, self.key_size)], axis=0)

        s, z, _, _ = lax.associative_scan(self.associative_update, (kv, key, start, next_done))

        # Discard prev state
        s = s[1:]
        z = z[1:]

        numer = jnp.einsum("tij, ti -> tj", s, query)
        denom = jnp.einsum("tzi, tj -> tz", z, query).clip(1e-6)

        output = numer / denom  
        output = self.ln(
            self.mlp(output) + self.skip(x)
        )

        return output, (s, z)

    def initial_state(self, shape=tuple()):
        return (
            jnp.zeros(
                (
                    *shape,
                    1,
                    self.key_size,
                    self.value_size,
                ),
                dtype=jnp.float32,
            ),
            jnp.zeros(
                (*shape, 1, 1, self.key_size), dtype=jnp.float32
            ),
        )