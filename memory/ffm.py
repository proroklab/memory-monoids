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

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.pre = eqx.filter_vmap(nn.Linear(input_size, trace_size, key=k1))
        self.gate_in = eqx.filter_vmap(Gate(input_size, trace_size, key=k2))
        self.gate_out = eqx.filter_vmap(Gate(input_size, self.output_size, key=k3))
        self.skip = eqx.filter_vmap(nn.Linear(input_size, self.output_size, key=k4))
        self.ffa_params = ffa.init_deterministic(trace_size, context_size)
        self.mix = eqx.filter_vmap(nn.Linear(2 * trace_size * context_size, self.output_size, key=k5))
        self.ln = eqx.filter_vmap(nn.LayerNorm(self.output_size, use_weight=False, use_bias=False))

    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        gate_in = self.gate_in(x)
        pre = self.pre(x)
        gated_x = pre * gate_in
        #state = partial(ffa.apply, self.ffa_params)(gated_x, state, start, next_done)
        #state = ffa.apply(self.ffa_params, gated_x, state, start, next_done)
        scan_input = jnp.repeat(jnp.expand_dims(gated_x, 2), self.context_size, axis=2)
        state = self.scan(scan_input, state, start)
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1).reshape(
            state.shape[0], -1
        )
        z = self.mix(z_in)
        gate_out = self.gate_out(x)
        skip = self.skip(x)
        out = self.ln(z * gate_out) + skip * (1 - gate_out)
        final_state = state[-1:]

        return out, final_state

    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)

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
        state = state * self.gamma(j) + x
        return state, j + i

    def wrapped_associative_update(self, carry, incoming): 
        prev_start, state, i = carry
        start, x, j = incoming
        # Reset all elements in the carry if we are starting a new episode
        state = state * jnp.logical_not(start) 
        j = j * jnp.logical_not(start)
        incoming = x, j
        carry = (state, i)
        out = self.unwrapped_associative_update(carry, incoming)
        start_out = jnp.logical_or(start, prev_start)
        return (start_out, *out)

    def scan(
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
        #timestep = jnp.arange(T + 1, dtype=jnp.int32)
        timestep = jnp.ones(T + 1, dtype=jnp.int32).reshape(-1, 1, 1)
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
