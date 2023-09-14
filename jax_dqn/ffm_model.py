from ffm import FFM
from typing import Any, Dict, List
import jax
import equinox as eqx
from equinox import nn
from jax import random, vmap
import jax.numpy as jnp
from modules import mish, ortho_linear, final_linear


class FFMQNetwork(eqx.Module):
    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: eqx.Module
    post: eqx.Module
    value: eqx.Module
    advantage: eqx.Module
    scale: eqx.Module
    trace_size: int
    context_size: int
    name: str = "FFM"

    def __init__(self, obs_shape, act_shape, config, key):
        self.config = config
        self.input_size = obs_shape
        self.output_size = act_shape
        keys = random.split(key, 8)
        pre = nn.Sequential(
            [ortho_linear(keys[1], obs_shape, config["mlp_size"]), mish]
        )
        self.pre = eqx.filter_vmap(pre)
        self.memory = FFM(
            input_size=config["mlp_size"],
            output_size=config["recurrent_size"],
            memory_size=config["recurrent_size"] // 4,
            context_size=4,
            key=keys[2],
        )
        self.trace_size = config["recurrent_size"] // 4
        self.context_size = 4
        post = nn.Sequential(
            [
                ortho_linear(
                    keys[3], self.config["recurrent_size"], self.config["mlp_size"],
                ),
                mish,
                ortho_linear(
                    keys[4], self.config["mlp_size"], self.config["mlp_size"], 
                ),
                mish,
            ]
        )
        self.post = eqx.filter_vmap(post)
        value = final_linear(keys[5], self.config["mlp_size"], 1, scale=0.01)
        self.value = eqx.filter_vmap(value)
        advantage = ortho_linear(keys[6], self.config["mlp_size"], self.output_size)
        self.advantage = eqx.filter_vmap(advantage)
        scale = final_linear(keys[7], self.config["mlp_size"], 1, scale=0.01)
        self.scale = eqx.filter_vmap(scale)

    
    @eqx.filter_jit
    def __call__(self, x, state, start, done, key):
        x = self.pre(x)
        #start = jnp.zeros_like(start)
        output, state = self.memory(x=x, state=state, start=start, key=key)
        y = self.post(output)

        value = self.value(y)
        A = self.advantage(y)
        scale = self.scale(y)

        A_normed = A / (1e-6 + jnp.linalg.norm(A, axis=-1, keepdims=True))
        advantage = A_normed - jnp.mean(A_normed, axis=-1, keepdims=True)
        # TODO: Only use target network for advantage branch
        # Let value/scale increase as needed
        q = value + scale * advantage
        return q, state

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1, self.trace_size, self.context_size), dtype=jnp.complex64)
