from typing import Any, Dict, Tuple
import jax
from jax import vmap, numpy as jnp
import equinox as eqx
from equinox import nn

import memory.ffa as ffa
from modules import complex_symlog, linsymlog, mish, softsymlog, symlog


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
    mag: nn.Linear
    phase: nn.Linear
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

        _, k1, k2, k3, k4 = jax.random.split(key, 5)
        self.pre = eqx.filter_vmap(nn.Linear(input_size, trace_size, key=k1))
        self.skip = eqx.filter_vmap(nn.Linear(input_size, self.output_size, key=k2))
        self.ffa_params = ffa.init(trace_size, context_size)
        self.mix = eqx.filter_vmap(nn.Linear(trace_size * context_size, self.output_size, key=k3))
        self.mag = eqx.filter_vmap(nn.Linear(trace_size * context_size, self.output_size, key=k3))
        self.phase = eqx.filter_vmap(nn.Linear(trace_size * context_size, self.output_size, key=k4))
        self.ln = eqx.filter_vmap(nn.LayerNorm(None, use_weight=False, use_bias=False))

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)

    # TODO: WTF why does adding another jit change performance...
    @eqx.filter_jit
    def __call__(
        self, x: jax.Array, state: jax.Array, start: jax.Array, next_done, key
    ) -> Tuple[jax.Array, jax.Array]:
        pre = self.pre(x)
        pre = pre / (1e-6 + jnp.linalg.norm(pre, axis=-1, keepdims=True, ord=2))
        state = ffa.apply(params=self.ffa_params, x=pre, state=state, start=start, next_done=next_done)
        s = state.reshape(state.shape[0], -1)
        mag = mish(self.mag(jnp.log(1 + jnp.abs(s))))
        phase = mish(self.phase(jnp.angle(s)))
        z = self.mix(mag * phase) + self.skip(x)
        
        final_state = state[-1:]
        return z, final_state


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
