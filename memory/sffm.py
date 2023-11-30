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
            jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)
            for _ in range(self.num_blocks)
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
        self.ffa_params = ffa.init(trace_size, context_size, k2)
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
        state = ffa.apply(params=self.ffa_params, x=pre, state=state, start=start, next_done=next_done)
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
