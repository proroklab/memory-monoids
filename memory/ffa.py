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
    a = jnp.linspace(a_low, a_high, memory_size)
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
    memory_size, context_size = len(a), len(b)
    a = jnp.clip(jnp.reshape(a, (1, memory_size, 1)), a_max=-1e-6)
    b = jnp.reshape(b, (1, 1, context_size))
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
    params: Tuple[jax.Array, jax.Array],
    carry: Tuple[jax.Array, jax.Array, jax.Array],
    incoming: Tuple[jax.Array, jax.Array, jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    state, i, _, done = carry
    x, j, start, next_done = incoming
    # VALIDATED: Only need start...
    #state = state * gamma(params, j - i) * jnp.logical_not(next_done) + x * jnp.logical_not(done)
    state = state * gamma(params, j - i) * jnp.logical_not(next_done) + x * jnp.logical_not(done)
    return state, j, start, next_done


def apply_one_ep(params, x, state, done):
    xs = []
    i = 0
    for i in range(len(done)):
        state = state * gamma(params, jnp.array([1])) + x[i:i+1]
        xs.append(state)
        if done[i]:
            break
        i += 1
    return jnp.concatenate(xs)


def check_apply(params, new_state, x, state, done):
    ep1 = apply_one_ep(params, x, jnp.zeros_like(state), done)
    idx = ep1.shape[0]
    ep2 = apply_one_ep(params, x[idx:], jnp.zeros_like(state), done[idx:])
    idx = idx + ep2.shape[0]
    ep3 = apply_one_ep(params, x[idx:], jnp.zeros_like(state), done[idx:])
    desired = jax.lax.stop_gradient(jnp.concatenate([ep1, ep2, ep3]))
    actual = jax.lax.stop_gradient(new_state[:desired.shape[0]])
    breakpoint()


# Verified fine again
@jax.jit
def apply(
    params: Tuple[jax.Array, jax.Array], x: jax.Array, state: jax.Array, start: jax.Array, next_done: jax.Array
) -> jax.Array:
    # x: [T, memory_size]
    # memory: [1, memory_size, context_size]
    timestep = jnp.arange(x.shape[0], dtype=jnp.complex64)
    # Add context dim
    x = jnp.expand_dims(x, axis=-1).astype(jnp.complex64)
    start = start.reshape(start.shape[0], 1, 1)
    next_done = next_done.reshape(next_done.shape[0], 1, 1)
    parameterized_update = partial(associative_update, params)

    # Fold the previous recurrent state into x (if not start)
    x = state * gamma(params, jnp.array([1])) + x
    #x = jnp.logical_not(next_done[0:1]) * state * gamma(params, jnp.array([1])) + x
    # This is not executed during inference -- method will just return x if size is 1
    new_state, _, _, _ = jax.lax.associative_scan(
        parameterized_update, (x, timestep, start, next_done), axis=0
    )
    if x.shape[0] > 1:
        check_apply(params, new_state, x, state, next_done)
    return new_state


if __name__ == "__main__":
    params = init(memory_size=2, context_size=3)
    x = jnp.ones(10, dtype=jnp.float32).reshape(5, 2)
    s = initial_state(params)
    start = jnp.zeros(5, dtype=bool)
    result = apply(params, x, s, start)
    breakpoint()
