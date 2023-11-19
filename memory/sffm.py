from typing import Any, Dict, Tuple
import jax
from jax import vmap, numpy as jnp
import equinox as eqx
from equinox import nn

import memory.ffa as ffa
from modules import final_linear, ortho_linear, symlog, complex_symlog, gaussian, leaky_relu, linear_softplus, mish, smooth_leaky_relu, soft_relglu, gelu, RandomSequential, default_init


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
        key: jax.random.PRNGKey,
    ):
        k = jax.random.split(key, num_blocks + 1)
        self.sffm = [
            SFFM(input_size, trace_size, context_size, k[i+1])
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
    

class LSEPool(eqx.Module):
    def __call__(self, x, key=None):
        context = jax.nn.logsumexp(x, axis=-1) 
        trace = jax.nn.logsumexp(x, axis=-2) 
        return jnp.concatenate([context, trace], axis=-1)

class MaxPool(eqx.Module):
    def __call__(self, x, key=None):
        abs_x = jnp.abs(x)
        context_idx = jnp.argmax(abs_x, axis=-1, keepdims=True)
        trace_idx = jnp.argmax(abs_x, axis=-2, keepdims=True)
        return jnp.concatenate([
            jnp.take_along_axis(x, context_idx, -1).squeeze(-1),
            jnp.take_along_axis(x, trace_idx, -2).squeeze(-2)
        ], axis=-1)
    

class MeanPool(eqx.Module):
    def __call__(self, x, key=None):
        context = jnp.mean(x, axis=-1) 
        trace = jnp.mean(x, axis=-2) 
        return jnp.concatenate([context, trace], axis=-1)

class DualAttention(eqx.Module):
    c: nn.Linear
    a: nn.Linear

    def __init__(self, input_size, trace_size, context_size, key):
        _, k0, k1 = jax.random.split(key, 3)
        self.a = eqx.filter_vmap(nn.Linear(input_size, trace_size, use_bias=False, key=k0))
        self.c = eqx.filter_vmap(nn.Linear(input_size, context_size, use_bias=False, key=k1))

    def __call__(self, x, state, key=None):
        a, c = self.a(x), self.c(x)
        a = jax.lax.complex(jax.nn.softmax(a, axis=-1), jnp.zeros_like(a))
        c = jax.lax.complex(jax.nn.softmax(c, axis=-1), jnp.zeros_like(c))
        a_attn = jnp.einsum("btc, bt -> bc", state, a)
        c_attn = jnp.einsum("btc, bc -> bt", state, c)
        return jnp.concatenate([a_attn, c_attn], axis=-1)


class DynamicLN(eqx.Module):
    def __call__(self, x, key=None):
        x = x - x.mean(axis=(-2,-1))
        std = jnp.clip(x.std(), a_min=1e-6, a_max=1.0)
        return x / std

class FroLinear(nn.Linear):
    weight: jax.Array
    bias: jax.Array

    def __call__(self, x, key=None):
        weight = self.weight / (1e-6 + jnp.linalg.norm(self.weight, ord='fro', keepdims=True))
        return weight @ x + self.bias

        

class SFFM(eqx.Module):
    input_size: int
    trace_size: int
    context_size: int
    name: str = "SFFM"

    ffa_params: Tuple[jax.Array, jax.Array]
    W_trace: nn.Linear
    W_context: nn.Linear
    mix: nn.Linear
    ln: nn.LayerNorm
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
        self.W_trace = eqx.filter_vmap(nn.Linear(input_size, trace_size, use_bias=False, key=k1))
        self.W_context = eqx.filter_vmap(nn.Linear(input_size, trace_size, use_bias=False, key=k8))
        # self.W_trace = eqx.filter_vmap(
        #     ortho_linear(k1, input_size, trace_size)
        # )
        # self.W_context = eqx.filter_vmap(
        #     ortho_linear(k8, input_size, context_size)
        # )
        self.ffa_params = ffa.init(trace_size, context_size, k2)
        self.mix = eqx.filter_vmap(RandomSequential([
            #nn.Linear(2 * trace_size * context_size, input_size, key=k5),
            #final_linear(k5, 2 * trace_size * context_size, input_size, 0.001),
            #ortho_linear(k5, 2 * trace_size * context_size, input_size),
            default_init(k5, nn.Linear(2 * trace_size * context_size, input_size, key=k5), scale=1e-4, zero_bias=True),
            nn.LayerNorm((input_size,), use_weight=False, use_bias=False),
            nn.Dropout(0.005),
            leaky_relu,
            nn.Linear(input_size, input_size, key=k6),
            #ortho_linear(k6, input_size, input_size),
            #nn.LayerNorm((input_size,)),
            nn.LayerNorm((input_size,), use_weight=False, use_bias=False),
            nn.Dropout(0.005),
            leaky_relu,
        ]))
        #self.ln = eqx.filter_vmap(nn.LayerNorm((input_size,)))
        self.ln = eqx.filter_vmap(nn.LayerNorm((input_size,), use_weight=False, use_bias=False))

    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)

    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        pre = jnp.abs(jnp.einsum("bi, bj -> bij", self.W_trace(x), self.W_context(x)))
        pre = pre / (1e-6 + jnp.sum(pre, axis=(-2,-1), keepdims=True))
        state = ffa.apply(params=self.ffa_params, x=pre, state=state, start=start, next_done=next_done)
        keys = jax.random.split(key, state.shape[0])
        s = state.reshape(state.shape[0], self.context_size * self.trace_size)
        scaled = jnp.concatenate([
            jnp.log(1 + jnp.abs(s)) * jnp.sin(jnp.angle(s)),
            jnp.log(1 + jnp.abs(s)) * jnp.cos(jnp.angle(s)),
        ], axis=-1)
        z = self.mix(scaled, keys)
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
