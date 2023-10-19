import math
from typing import Tuple
import jax
from jax import numpy as jnp
from functools import partial


def init(
    memory_size: int, context_size: int, min_period: int = 1, max_period: int = 1024
) -> Tuple[jax.Array, jax.Array]:
    a_low = -math.e
    a_high = -1e-6
    #a = jnp.linspace(a_low, a_high, memory_size)
    a = jnp.linspace(-0.5, 0.0, memory_size)
    b = 2 * jnp.pi / jnp.linspace(min_period, max_period, context_size)
    return a, b


@jax.jit
def initial_state(params: Tuple[jax.Array, jax.Array]) -> jax.Array:
    a, b = params
    memory_size, context_size = len(a), len(b)
    return jnp.zeros((1, memory_size, context_size))


@jax.jit
def log_gamma(params: Tuple[jax.Array, jax.Array], t: jax.Array) -> jax.Array:
    a, b = params
    memory_size, context_size = a.shape[-1], b.shape[-1]
    # a = jnp.clip(jnp.reshape(a, (1, memory_size, 1)), a_max=-1e-6)
    # b = jnp.reshape(b, (1, 1, context_size))
    a = jnp.clip(jnp.reshape(a, (t.shape[0], memory_size, 1)), a_max=-1e-6)
    b = jnp.clip(jnp.reshape(b, (t.shape[0], 1, context_size)), a_max=2 * jnp.pi - 1e-6)
    ab = jax.lax.complex(a, b)
    return ab * t.reshape(t.shape[0], 1, 1)


@jax.jit
def gamma(params: Tuple[jax.Array, jax.Array], t: jax.Array) -> jax.Array:
    return jnp.exp(log_gamma(params, t))


# Currently, next_done points to the final obs in an episode
# and start points to the initial obs in an episode
# obs:       [0, 1, 1, 0, 1] (here the zero is the initial obs of an episode)
# start:     [1, 0, 0, 1, 0]
# prev_start:[0, 1, 0, 0, 1]
# done:      [0, 0, 0, 1, 0]
# next_done: [0, 0, 1, 0, 0]
# During inference, "next_done" will never be received because we only
# ever see one step/episode at a time.
# This is probably what we want (to only use next_done during training)
@jax.jit
def associative_update(
    #params: Tuple[jax.Array, jax.Array],
    carry: Tuple[jax.Array, jax.Array, jax.Array],
    incoming: Tuple[jax.Array, jax.Array, jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    _, state, i, prev_start, done = carry
    params, x, j, start, next_done = incoming
    state = jnp.logical_not(start) * state * gamma(params, j - i) + x
    return params, state, j, jnp.logical_or(start, prev_start), next_done


# Verified fine again
@jax.jit
def apply(
    params: Tuple[jax.Array, jax.Array], x: jax.Array, state: jax.Array, start: jax.Array, next_done: jax.Array
) -> jax.Array:
    # x: [T, memory_size]
    # memory: [1, memory_size, context_size]
    T = x.shape[0]
    memory_size, context_size = len(params[0]), len(params[1])
    timestep = jnp.arange(T + 1, dtype=jnp.int32)
    # Add context dim
    x = jnp.repeat(jnp.expand_dims(x, axis=-1).astype(jnp.complex64), context_size, axis=-1)
    start = start.reshape(T, 1, 1)
    next_done = next_done.reshape(T, 1, 1)

    # Now insert previous recurrent state
    x = jnp.concatenate([state, x], axis=0)
    start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)
    next_done = jnp.concatenate([jnp.zeros_like(next_done[:1]), next_done], axis=0)

    broadcasted_params = [jnp.broadcast_to(jnp.expand_dims(p, 0), (x.shape[0], *p.shape)) for p in params]

    # Fold the previous recurrent state into x (if not start)
    #x = state * gamma(params, jnp.array([1])) + x
    # This is not executed during inference -- method will just return x if size is 1
    _, new_state, _, _, _ = jax.lax.associative_scan(
        #parameterized_update, (x, timestep, start, next_done), axis=0
        associative_update, (broadcasted_params, x, timestep, start, next_done), axis=0
    )
    return new_state[1:]


if __name__ == "__main__":
    params = init(memory_size=2, context_size=3)
    x = jnp.ones(10, dtype=jnp.float32).reshape(5, 2)
    s = initial_state(params)
    start = jnp.zeros(5, dtype=bool)
    result = apply(params, x, s, start)
    breakpoint()
