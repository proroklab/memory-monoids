from typing import Any, Callable, Dict, List, Tuple
import equinox as eqx
from equinox import nn
import jax
from jax import random, vmap, lax
import jax.numpy as jnp
from memory.module import MemoryModule
from modules import Lambda, leaky_relu
from utils import expand_right


def phi(x, key=None):
    return 1 + jax.nn.elu(x)


class NLinearAttention(eqx.Module):
    key_size: int
    value_size: int
    num_blocks: int
    blocks: List[eqx.Module]
    name: str = "NLinearAttention"


    def __init__(
        self,
        input_size: int,
        key_size: int,
        value_size: int,
        num_blocks: int,
        key: jax.random.PRNGKey,
    ):
        k = jax.random.split(key, num_blocks + 1)
        self.blocks = [
            LinearAttention(input_size, key_size, value_size, k[i+1])
            for i in range(num_blocks)
        ]
        self.key_size = key_size
        self.value_size = value_size
        self.num_blocks = num_blocks

    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        for i, block in enumerate(self.sffm):
            key, k = jax.random.split(key)
            y, s = block(x, state[i], start, next_done, k)
            x = x + y
            state[i] = s
        return y, state 

    def initial_state(self, shape=tuple()):
        return [
            s.initial_state(shape) for s in self.blocks
        ]

class LinearAttention(MemoryModule):
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
                leaky_relu,
                nn.Linear(hidden_size, hidden_size, key=keys[5]),
                leaky_relu,
                nn.Linear(hidden_size, hidden_size, key=keys[6]),
            ]
        ))

        self.ln = eqx.filter_vmap(nn.LayerNorm(hidden_size))

    def associative_update(self, carry, inputs):
        s, z = carry
        kv, k = inputs

        s = s + kv
        z = z + k

        return (s, z)

    def wrapped_associative_update(self, carry: jax.Array, incoming: jax.Array) -> Tuple[jax.Array, ...]:
        """The reset-wrapped form of the associative update. 

        You might need to override this
        if you use variables in associative_update that are not from initial_state. 
        This is equivalent to the h function in the paper:
        b x H -> b x H
        """
        prev_start, *carry = carry
        start, *incoming = incoming
        # Reset all elements in the carry if we are starting a new episode
        s, z = carry

        s = jnp.logical_not(start) * s
        z = jnp.logical_not(start) * z

        out = self.associative_update((s, z), incoming)
        start_out = jnp.logical_or(start, prev_start)
        return (start_out, *out)

    def scan(
        self,
        x: jax.Array,
        state: List[jax.Array],
        start: jax.Array,
    ) -> jax.Array:
        """Given an input and recurrent state, this will update the recurrent state. This is equivalent
        to the inner-function g in the paper:
        g: O x H -> S
        """
        # x: [T, ...]
        # memory: [1, ...]
        kv, k = x
        s, z = state
        T = k.shape[0]
        start = start.reshape([T, 1, 1])

        # Now insert previous recurrent state
        s = jnp.concatenate([s, kv], axis=0)
        z = jnp.concatenate([z, k], axis=0)
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)

        # This is not executed during inference -- method will just return x if size is 1
        _, s, z = lax.associative_scan(
            self.wrapped_associative_update,
            (start, s, z),
            axis=0,
        )
        return s[1:], z[1:]

    def __call__(self, x, state, start, next_done, key=None):
        s, z = state

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        kv = jnp.einsum("ti, tj -> tij", key, value)

        s, z = self.scan((kv, key.reshape(key.shape[0], 1, -1)), state, start)

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