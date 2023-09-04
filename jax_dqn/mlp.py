import math
from typing import Any, Dict
import jax
import equinox as eqx
from equinox import nn
from jax import random, vmap
import jax.numpy as jnp
from modules import mish, FinalLinear, NoisyLinear, final_layer_init


class MLPQNetwork(eqx.Module):
    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    post: eqx.Module
    name: str = "MLP"

    def __init__(self, obs_shape, act_shape, config, key):
        self.config = config
        self.input_size = obs_shape
        self.output_size = act_shape
        keys = random.split(key, 4)
        self.pre = nn.Sequential(
            [NoisyLinear(obs_shape, config["mlp_size"], init_std=config['noise_std'], key=keys[0]), mish]
        )
        self.post = nn.Sequential(
            [
                NoisyLinear(
                    self.config["mlp_size"], self.config["mlp_size"], init_std=config['noise_std'], key=keys[2]
                ),
                mish,
                #nn.Dropout(p=0.05),
                #NoisyLinear(self.config["mlp_size"], self.config["mlp_size"], key=keys[1]), mish,
                #nn.LayerNorm(self.config["mlp_size"], use_weight=False, use_bias=False),
                final_layer_init(NoisyLinear(
                    self.config["mlp_size"], self.output_size, init_std=config['noise_std'], key=keys[3]
                ))
            ]
        )

    @eqx.filter_jit
    def __call__(self, x, state, start, done, key):
        # key, pre_key, post_key = random.split(
        #     key,
        #     (3, math.prod(x.shape[:-1]))
        # )
        # TODO: key should be the same for the whole batch
        _, pre_key, post_key = random.split(key, 3)
        pre_key = jnp.tile(jnp.expand_dims(pre_key, 0), (x.shape[0], 1))
        post_key = jnp.tile(jnp.expand_dims(post_key, 0), (x.shape[0], 1))
        x = vmap(self.pre)(x, key=pre_key)
        y = vmap(self.post)(x, key=post_key)
        return y, jnp.zeros(1)

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, 1))
