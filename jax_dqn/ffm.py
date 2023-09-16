from typing import Any, Dict, Tuple
import jax
from jax import vmap, numpy as jnp
import equinox as eqx
from equinox import nn
from functools import partial

import ffa


class Gate(eqx.Module):
    linear: nn.Linear

    def __init__(self, input_size, output_size, key):
        self.linear = nn.Linear(input_size, output_size, key=key)

    def __call__(self, x):
        return jax.nn.sigmoid(self.linear(x))


class FFM(eqx.Module):
    input_size: int
    memory_size: int
    context_size: int
    output_size: int

    ffa_params: Tuple[jax.Array, jax.Array]
    ffa_init_kwargs: Dict[str, Any]

    pre: nn.Linear
    gate_in: Gate
    gate_out: Gate
    skip: nn.Linear
    mix: nn.Linear
    ln: nn.Linear

    def __init__(
        self,
        input_size: int,
        output_size: int,
        memory_size: int,
        context_size: int,
        key: jax.random.PRNGKey,
        ffa_init_kwargs: Dict[str, Any] = {},
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.context_size = context_size

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.pre = nn.Linear(input_size, memory_size, key=k1)
        self.gate_in = Gate(input_size, memory_size, key=k2)
        self.gate_out = Gate(input_size, output_size, key=k3)
        self.skip = nn.Linear(input_size, output_size, key=k4)
        self.ffa_params = ffa.init(memory_size, context_size)
        self.mix = nn.Linear(2 * memory_size * context_size, output_size, key=k5)
        self.ln = nn.LayerNorm(output_size, use_weight=False, use_bias=False)
        self.ffa_init_kwargs = ffa_init_kwargs

    def initial_state(self):
        return ffa.initial_state(self.ffa_params, **self.ffa_init_kwargs)

    @jax.jit
    @eqx.filter_jit
    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        #gate_in = vmap(self.gate_in)(x)
        pre = vmap(self.pre)(x)
        pre = pre / (1e-6 + jnp.linalg.norm(pre, axis=-1, keepdims=True))
        #gated_x = pre / jnp.linalg.norm(pre, axis=-1, keepdims=True) * gate_in
        #pre = vmap(jax.nn.relu)(vmap(self.pre)(x))
        #gated_x = gated_x / jnp.linalg.norm(gated_x, axis=-1, keepdims=True)
        state = partial(ffa.apply, self.ffa_params)(pre, state, start, next_done)
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1).reshape(
            state.shape[0], -1
        )
        z = vmap(self.mix)(z_in)
        #z = z - z.mean(axis=-1, keepdims=True)
        #z = z / (1e-6 + jnp.var(z, axis=-1, keepdims=True))
        #gate_out = vmap(self.gate_out)(x)
        skip = vmap(self.skip)(x)
        return z + skip, state
        #out = vmap(self.ln)(z * gate_out) + skip * (1 - gate_out)

        return out, state


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
