from typing import Tuple
import jax
from jax import numpy as jnp
import equinox as eqx
from equinox import nn

import memory.ffa as ffa
from modules import leaky_relu, RandomSequential, default_init


class NSFFM(eqx.Module):
    sffm: list
    trace_size: int
    context_size: int
    num_blocks: int
    name: str = "NSFFM"

    def __init__(
        self,
        input_size: int,
        trace_size: int,
        context_size: int,
        num_blocks: int,
        dropout: float,
        key: jax.random.PRNGKey,
    ):
        k = jax.random.split(key, num_blocks + 1)
        self.sffm = [
            SFFM(input_size, trace_size, context_size, dropout, k[i+1])
            for i in range(num_blocks)
        ]
        self.trace_size = trace_size
        self.context_size = context_size
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
            s.initial_state(shape) for s in self.sffm
        ]


class SFFM(eqx.Module):
    input_size: int
    trace_size: int
    context_size: int
    name: str = "SFFM"

    ffa_params: Tuple[jax.Array, jax.Array]
    W_trace: nn.Linear
    W_context: nn.Linear
    block0: nn.Linear
    block1: nn.Linear

    def __init__(
        self,
        input_size: int,
        trace_size: int,
        context_size: int,
        dropout: float,
        key: jax.random.PRNGKey,
    ):
        self.input_size = input_size
        self.trace_size = trace_size
        self.context_size = context_size

        k0, k1, k2, k3, k4 = jax.random.split(key, 5)
        self.W_trace = eqx.filter_vmap(nn.Linear(input_size, trace_size, use_bias=False, key=k0))
        self.W_context = eqx.filter_vmap(nn.Linear(input_size, trace_size, use_bias=False, key=k1))
        self.ffa_params = self.init_ffa(trace_size, context_size, key=k2)
        self.block0 = eqx.filter_vmap(RandomSequential([
            default_init(k3, nn.Linear(2 * trace_size * context_size, input_size, key=k3), scale=0.01),
            nn.LayerNorm((input_size,)),
            nn.Dropout(dropout),
            leaky_relu,
        ]))
        self.block1 = eqx.filter_vmap(RandomSequential([
            nn.Linear(input_size, input_size, key=k4),
            nn.LayerNorm((input_size,)),
            nn.Dropout(dropout),
            leaky_relu,
        ]))

    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)

    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        pre = jnp.abs(jnp.einsum("bi, bj -> bij", self.W_trace(x), self.W_context(x)))
        pre = pre / (1e-8 + jnp.sum(pre, axis=(-2,-1), keepdims=True))
        state = self.aggregate(pre, state, start)
        keys = jax.random.split(key, 2 * state.shape[0])
        s = state.reshape(state.shape[0], self.context_size * self.trace_size)
        scaled = jnp.concatenate([
            jnp.log(1 + jnp.abs(s)) * jnp.sin(jnp.angle(s)),
            jnp.log(1 + jnp.abs(s)) * jnp.cos(jnp.angle(s)),
        ], axis=-1)
        z = self.block0(scaled, keys[:state.shape[0]])
        z = self.block1(z, keys[state.shape[0]:])
        final_state = state[-1:]
        return z, final_state

    def init_ffa(
        self, memory_size: int, context_size: int, min_period: int = 1, max_period: int = 10_000, *, key
    ) -> Tuple[jax.Array, jax.Array]:
        _, k1, k2 = jax.random.split(key, 3)
        a_low = 1e-6
        a_high = 0.1
        a = jax.random.uniform(k1, (memory_size,), minval=a_low, maxval=a_high)
        b = 2 * jnp.pi / jnp.exp(jax.random.uniform(k2, (context_size,), minval=jnp.log(min_period), maxval=jnp.log(max_period)))
        return a, b

    def log_gamma(self, t: jax.Array) -> jax.Array:
        a, b = self.ffa_params
        a = -jnp.abs(a).reshape((1, self.trace_size, 1))
        b = b.reshape(1, 1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(t.shape[0], 1, 1)

    def gamma(self, t: jax.Array) -> jax.Array:
        return jnp.exp(self.log_gamma(t))

    def unwrapped_associative_update(
        self,
        carry: Tuple[jax.Array, jax.Array, jax.Array],
        incoming: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        state, i, = carry
        x, j = incoming
        state = state * self.gamma(j - i) + x
        return state, j, 

    def wrapped_associative_update(self, carry, incoming): 
        prev_start, state, i = carry
        start, x, j = incoming
        incoming = x, j
        # Reset all elements in the carry if we are starting a new episode
        state = state * jnp.logical_not(start) 
        carry = (state, i)
        out = self.unwrapped_associative_update(carry, incoming)
        start_out = jnp.logical_or(start, prev_start)
        return (start_out, *out)

    def aggregate(
        self,
        x: jax.Array,
        state: jax.Array,
        start: jax.Array,
    ) -> jax.Array:
        """Given an input and recurrent state, this will update the recurrent state. This is equivalent
        to the inner-function g in the paper."""
        # x: [T, memory_size]
        # memory: [1, memory_size, context_size]
        T = x.shape[0]
        timestep = jnp.arange(T + 1, dtype=jnp.int32)
        # Add context dim
        start = start.reshape(T, 1, 1)

        # Now insert previous recurrent state
        x = jnp.concatenate([state, x], axis=0)
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)

        # This is not executed during inference -- method will just return x if size is 1
        _, new_state, _ = jax.lax.associative_scan(
            self.wrapped_associative_update,
            (start, x, timestep),
            axis=0,
        )
        return new_state[1:]


if __name__ == "__main__":
    m = SFFM(
        input_size=2,
        trace_size=5,
        context_size=6,
        key=jax.random.PRNGKey(0),
    )
    s = m.initial_state()
    x = jnp.ones((10, 2))
    start = jnp.zeros(10, dtype=bool)
    out = m(x, s, start)
