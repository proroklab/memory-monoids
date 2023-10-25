from typing import Any, Dict, Tuple
import jax
from jax import vmap, numpy as jnp
import equinox as eqx
from equinox import nn

import memory.ffa as ffa
from modules import complex_symlog, gaussian, leaky_relu, linear_softplus, mish, smooth_leaky_relu


class NormalizedLinear(eqx.Module):
    linear: nn.Linear
    def __init__(self, input_size, output_size, key):
        self.linear = nn.Linear(input_size, output_size, key=key)
    
    def __call__(self, x, key=None):
        out = self.linear(x)
        return out / (1e-6 + jnp.linalg.norm(out, keepdims=True, ord=1))


class SFFM(eqx.Module):
    input_size: int
    trace_size: int
    context_size: int
    output_size: int
    name: str = "SFFM"

    ffa_params: Tuple[jax.Array, jax.Array]
    pre: nn.Linear
    skip: nn.Linear
    mix: nn.Linear
    ln: nn.LayerNorm

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

        _, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 9)
        self.pre = eqx.filter_vmap(nn.Sequential([
            nn.LayerNorm((None,), use_weight=False, use_bias=False),
            nn.Linear(input_size, trace_size, key=k1),
            linear_softplus,
        ]))
        self.skip = eqx.filter_vmap(nn.Linear(input_size, self.output_size, key=k2))
        self.ffa_params = ffa.init(trace_size, context_size)
        self.mix = eqx.filter_vmap(nn.Sequential([
            nn.Linear(2 * trace_size * context_size, trace_size * context_size, key=k5),
            mish,
            nn.Linear(trace_size * context_size, trace_size * context_size, key=k6),
            mish,
            nn.Linear(trace_size * context_size, self.output_size, key=k7)
        ])
        )
        self.ln = eqx.filter_vmap(nn.LayerNorm((None,), use_weight=False, use_bias=False))

    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)

    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        pre = self.pre(x)
        state = ffa.apply(params=self.ffa_params, x=pre, state=state, start=start, next_done=next_done)
        s = state.reshape(state.shape[0], -1)
        #scaled = s / (1e-6 + jnp.linalg.norm(s, ord=jnp.inf, axis=1, keepdims=True))
        scaled = (s - jnp.mean(s, keepdims=True)) / (1e-6 + jnp.std(s, keepdims=True))
        #scaled = self.ln(s)
        z = self.mix(jnp.concatenate([scaled.real, scaled.imag], axis=-1))
        final_state = state[-1:]
        return self.ln(z + self.skip(x)), final_state


if __name__ == "__main__":
    m = SFFM(
        input_size=2,
        output_size=4,
        trace_size=5,
        context_size=6,
        key=jax.random.PRNGKey(0),
    )
    s = m.initial_state()
    x = jnp.ones((10, 2))
    start = jnp.zeros(10, dtype=bool)
    out = m(x, s, start)
