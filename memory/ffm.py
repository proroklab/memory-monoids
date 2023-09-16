from typing import Any, Dict, Tuple
import jax
from jax import vmap, numpy as jnp
import equinox as eqx
from equinox import nn
from functools import partial

import memory.ffa as ffa


class Gate(eqx.Module):
    linear: nn.Linear

    def __init__(self, input_size, output_size, key):
        self.linear = nn.Linear(input_size, output_size, key=key)

    def __call__(self, x):
        return jax.nn.sigmoid(self.linear(x))


class FFM(eqx.Module):
    input_size: int
    trace_size: int
    context_size: int
    output_size: int
    name: str = "FFM"

    ffa_params: Tuple[jax.Array, jax.Array]
    pre: nn.Linear
    gate_in: Gate
    gate_out: Gate
    skip: nn.Linear
    mix: nn.Linear
    ln: nn.Linear

    def __init__(
        self,
        input_size: int,
        trace_size: int,
        context_size: int,
        output_size: int,
        key: jax.random.PRNGKey,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.trace_size = trace_size
        self.context_size = context_size

        _, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        self.pre = nn.Linear(input_size, trace_size, key=k1)
        self.gate_in = Gate(input_size, trace_size, key=k2)
        self.gate_out = Gate(input_size, self.output_size, key=k3)
        self.skip = nn.Linear(input_size, self.output_size, key=k4)
        self.ffa_params = ffa.init(trace_size, context_size)
        self.mix = nn.Linear(2 * trace_size * context_size, self.output_size, key=k5)
        self.ln = nn.LayerNorm(self.output_size, use_weight=False, use_bias=False)

    @eqx.filter_jit
    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        gate_in = vmap(self.gate_in)(x)
        pre = vmap(self.pre)(x)
        gated_x = pre * gate_in
        state = partial(ffa.apply, self.ffa_params)(gated_x, state, start, next_done)
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1).reshape(
            state.shape[0], -1
        )
        z = vmap(self.mix)(z_in)
        gate_out = vmap(self.gate_out)(x)
        skip = vmap(self.skip)(x)
        out = vmap(self.ln)(z * gate_out) + skip * (1 - gate_out)
        final_state = state[-1:]

        return out, final_state

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)


if __name__ == "__main__":
    m = FFM(
        input_size=2,
        output_size=4,
        memory_size=5,
        context_size=6,
        key=jax.random.PRNGKey(0),
    )
    s = m.initial_state()
    x = jnp.ones((10, 2))
    start = jnp.zeros(10, dtype=bool)
    out = m(x, s, start)
