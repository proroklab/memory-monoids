from typing import Any, Dict, Tuple
import jax
from jax import vmap, numpy as jnp
import equinox as eqx
from equinox import nn

import memory.ffa as ffa
from modules import complex_symlog, gaussian, leaky_relu, linear_softplus, mish, smooth_leaky_relu, soft_relglu


class NormalizedLinear(eqx.Module):
    linear: nn.Linear
    def __init__(self, input_size, output_size, key):
        self.linear = nn.Linear(input_size, output_size, key=key)
    
    def __call__(self, x, key=None):
        out = self.linear(x)
        return out / (1e-6 + jnp.linalg.norm(out, keepdims=True, ord=1))


def super_glu(x, key=None):
    return jnp.tanh(jax.nn.glu(x))

def relglu(x, key=None):
    return jax.nn.relu(jax.nn.glu(x))

def relsglu(x, key=None):
    a, b = jnp.split(x, 2, -1)
    return jax.nn.relu(a) * jax.nn.softmax(b, axis=-1)

def expsglu(x, key=None):
    a, b = jnp.split(x, 2, -1)
    return jnp.exp(a) * jax.nn.softmax(b, axis=-1)

def elu_glu(x, key=None):
    a, b = jnp.split(x, 2, -1)
    return (1 + jax.nn.elu(a)) * jax.nn.sigmoid(b)


class DSFFM(eqx.Module):
    sffm0: eqx.Module
    sffm1: eqx.Module
    trace_size: int
    context_size: int
    name: str = "DSFFM"

    def __init__(
        self,
        input_size: int,
        trace_size: int,
        context_size: int,
        key: jax.random.PRNGKey,
    ):
        _, k1, k2 = jax.random.split(key, 3)
        self.sffm0 = SFFM(input_size, trace_size, context_size, k1)
        self.sffm1 = SFFM(input_size, trace_size, context_size, k2)
        self.trace_size = trace_size
        self.context_size = context_size

    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        _, k1, k2 = jax.random.split(key, 3)
        s0, s1 = state
        y0, s0 = self.sffm0(x, s0, start, next_done, k1)
        y1, s1 = self.sffm1(y0, s1, start, next_done, k2)
        return y1, (s0, s1)

    def initial_state(self, shape=tuple()):
        return [
            jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)
            for _ in range(2)
        ]
    
class SFFM(eqx.Module):
    input_size: int
    trace_size: int
    context_size: int
    name: str = "SFFM"

    ffa_params: Tuple[jax.Array, jax.Array]
    pre: nn.Linear
    mix: nn.Linear
    ln: nn.LayerNorm
    ln2: nn.LayerNorm
    drop: nn.Dropout

    def __init__(
        self,
        input_size: int,
        trace_size: int,
        context_size: int,
        key: jax.random.PRNGKey,
    ):
        self.input_size = input_size
        self.trace_size = trace_size
        self.context_size = context_size

        _, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 9)
        self.pre = eqx.filter_vmap(nn.Linear(input_size, trace_size, key=k1))
        self.ffa_params = ffa.init(trace_size, context_size, k2)
        self.mix = eqx.filter_vmap(nn.Sequential([
            nn.Linear(2 * trace_size * context_size, input_size, key=k5),
            nn.LayerNorm((input_size,), use_weight=False, use_bias=False),
            mish,
            nn.Linear(input_size, input_size, key=k6),
            nn.LayerNorm((input_size,), use_weight=False, use_bias=False),
            mish,
            nn.Linear(input_size, input_size, key=k7)
        ])
        )
        self.ln = eqx.filter_vmap(nn.LayerNorm((input_size,), use_weight=False, use_bias=False))
        self.ln2 = eqx.filter_vmap(nn.LayerNorm((context_size * trace_size,), use_weight=False, use_bias=False))
        self.drop = nn.Dropout(0.05)

    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)

    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        pre = self.pre(x)
        state = ffa.apply(params=self.ffa_params, x=pre, state=state, start=start, next_done=next_done)
        s = state.reshape(state.shape[0], -1)
        scaled = self.ln2(s)
        z = self.mix(jnp.concatenate([scaled.real, scaled.imag], axis=-1))
        final_state = state[-1:]
        return self.ln(z + x), final_state


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
