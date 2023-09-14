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


@jax.jit
def associative_update(
    params: Tuple[jax.Array, jax.Array],
    carry: Tuple[jax.Array, jax.Array, jax.Array],
    incoming: Tuple[jax.Array, jax.Array, jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    state, i, _ = carry
    x, j, start = incoming
    state = jnp.logical_not(start) * state * gamma(params, j - i) + x
    return state, j, start


@jax.jit
def apply(
    params: Tuple[jax.Array, jax.Array], x: jax.Array, state: jax.Array, start: jax.Array
) -> jax.Array:
    # x: [T, memory_size]
    # memory: [1, memory_size, context_size]
    a, b = params
    memory_size, context_size = len(a), len(b)

    timestep = jnp.arange(x.shape[0] + 1, dtype=jnp.float32)
    timestep = jax.lax.complex(timestep, jnp.zeros_like(timestep))

    # [prev_state, x_0, x_1, ...]
    x = jnp.repeat(x.reshape(*x.shape, 1), context_size, axis=-1)
    x = jax.lax.complex(x, jnp.zeros_like(x))
    x = jnp.concatenate([state, x], axis=0)

    # [False, ...]
    start = jnp.concatenate([jnp.array([False]), start], axis=0)
    start = start.reshape(start.shape[0], 1, 1)
    
    parameterized_update = partial(associative_update, params)
    state, _, _ = jax.lax.associative_scan(
        parameterized_update, (x, timestep, start), axis=0
    )
    return state[1:]


if __name__ == "__main__":
    params = init(memory_size=2, context_size=3)
    x = jnp.ones(10, dtype=jnp.float32).reshape(5, 2)
    s = initial_state(params)
    start = jnp.zeros(5, dtype=bool)
    result = apply(params, x, s, start)
    breakpoint()
