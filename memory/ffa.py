import math
from typing import Tuple
import jax
from jax import numpy as jnp
from functools import partial



def init(
    memory_size: int, context_size: int, key, min_period: int = 1, max_period: int = 10_000 
) -> Tuple[jax.Array, jax.Array]:
    _, k1, k2 = jax.random.split(key, 3)
    a_low = 1e-6
    a_high = 0.1
    a = jax.random.uniform(k1, (memory_size,), minval=a_low, maxval=a_high)
    b = 2 * jnp.pi / jnp.exp(jax.random.uniform(k2, (context_size,), minval=jnp.log(min_period), maxval=jnp.log(max_period)))
    return a, b


def initial_state(params: Tuple[jax.Array, jax.Array]) -> jax.Array:
    a, b = params
    memory_size, context_size = len(a), len(b)
    return jnp.zeros((1, memory_size, context_size))


def log_gamma(params: Tuple[jax.Array, jax.Array], t: jax.Array) -> jax.Array:
    a, b = params
    memory_size, context_size = a.shape[-1], b.shape[-1]
    #a = jnp.clip(jnp.reshape(a, (t.shape[0], memory_size, 1)), a_max=0)
    a = -jnp.abs(a).reshape((t.shape[0], memory_size, 1))
    b = b.reshape(t.shape[0], 1, context_size)
    ab = jax.lax.complex(a, b)
    return ab * t.reshape(t.shape[0], 1, 1)


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
def associative_update(
    carry: Tuple[jax.Array, jax.Array, jax.Array],
    incoming: Tuple[jax.Array, jax.Array, jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    _, state, i, prev_start, done = carry
    params, x, j, start, next_done = incoming
    prev_state = jnp.logical_not(start) * state + start * jnp.zeros_like(state)
    state = prev_state * gamma(params, j - i) + x
    return params, state, j, jnp.logical_or(start, prev_start), next_done

# Verified fine again
def apply(
    params: Tuple[jax.Array, jax.Array],
    x: jax.Array,
    state: jax.Array,
    start: jax.Array,
    next_done: jax.Array,
) -> jax.Array:
    # x: [T, memory_size]
    # memory: [1, memory_size, context_size]
    T = x.shape[0]
    memory_size, context_size = len(params[0]), len(params[1])
    timestep = jnp.arange(T + 1, dtype=jnp.int32)
    # Add context dim
#    x = jnp.repeat(
#        jnp.expand_dims(x, axis=-1).astype(jnp.complex64), context_size, axis=-1
#    )
    start = start.reshape(T, 1, 1)
    next_done = next_done.reshape(T, 1, 1)

    # Now insert previous recurrent state
    x = jnp.concatenate([state, x], axis=0)
    start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)
    next_done = jnp.concatenate([jnp.zeros_like(next_done[:1]), next_done], axis=0)

    broadcasted_params = [
        jnp.broadcast_to(jnp.expand_dims(p, 0), (x.shape[0], *p.shape)) for p in params
    ]

    # Fold the previous recurrent state into x (if not start)
    # x = state * gamma(params, jnp.array([1])) + x
    # This is not executed during inference -- method will just return x if size is 1
    _, new_state, _, _, _ = jax.lax.associative_scan(
        # parameterized_update, (x, timestep, start, next_done), axis=0
        associative_update,
        (broadcasted_params, x, timestep, start, next_done),
        axis=0,
    )
    return new_state[1:]


if __name__ == "__main__":
    params = init(memory_size=2, context_size=3)
    x = jnp.ones(10, dtype=jnp.float32).reshape(5, 2)
    s = initial_state(params)
    start = jnp.zeros(5, dtype=bool)
    result = apply(params, x, s, start)
    breakpoint()
