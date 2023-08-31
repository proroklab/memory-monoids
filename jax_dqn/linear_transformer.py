from typing import Any, Callable, Dict, List
import equinox as eqx
from equinox import nn
import jax
from jax import random, vmap, lax
import jax.numpy as jnp
from modules import Lambda
from modules import mish, FinalLinear
from utils import expand_right


class LinearAttention(eqx.Module):
    key: eqx.Module
    value: eqx.Module
    query: eqx.Module
    mlp: eqx.Module
    phi: Callable
    ln: eqx.Module
    skip: eqx.Module

    def __init__(self, input_dim, hidden_dim, key):
        keys = random.split(key, 7)
        self.phi = jax.nn.elu
        self.key = nn.Sequential(
            [
                nn.Linear(input_dim, hidden_dim, use_bias=False, key=keys[0]),
                Lambda(self.phi),
            ]
        )
        self.query = nn.Sequential(
            [
                nn.Linear(input_dim, hidden_dim, use_bias=False, key=keys[1]),
                Lambda(self.phi),
            ]
        )
        self.value = nn.Linear(input_dim, hidden_dim, use_bias=False, key=keys[2])
        self.skip = nn.Linear(input_dim, hidden_dim, key=keys[3])
        self.mlp = nn.Sequential(
            [
                nn.Linear(hidden_dim, hidden_dim, key=keys[4]),
                mish,
                nn.Linear(hidden_dim, hidden_dim, key=keys[5]),
                mish,
                nn.Linear(hidden_dim, hidden_dim, key=keys[6]),
            ]
        )

        self.ln = nn.LayerNorm(hidden_dim)

    def binary_op(self, carry, inputs):
        kv, k, start, done = inputs
        _s, _z, _, _ = carry
        s = _s * expand_right(start, _s.shape) + kv * expand_right(done, kv.shape)
        z = _z * expand_right(start, _z.shape) + k * expand_right(done, k.shape)
        if z.shape[1] > 1:
            breakpoint()
        return s, z, start, done

    def binary_s_op(self, carry, inputs):
        kv, start, done = inputs
        s, _, _ = carry
        s = s * expand_right(start, s.shape) + kv * expand_right(done, kv.shape)
        return s, start, done

    def binary_z_op(self, carry, inputs):
        k, start, done = inputs
        z, _, _ = carry
        z = z * expand_right(start, z.shape) + k * expand_right(done, k.shape)
        return z, start, done

    def __call__(self, x, state, start, done):
        s, z = state
        key = eqx.filter_vmap(self.key)(x)
        query = eqx.filter_vmap(self.query)(x)
        value = eqx.filter_vmap(self.value)(x)
        start = start.reshape(*start.shape, 1)
        done = done.reshape(*done.shape, 1)

        kv = jnp.einsum("ti, tj -> tij", key, value)
        kv = jnp.concatenate([s, kv], axis=0)
        key = jnp.concatenate([z, key], axis=0)

        # print(s.shape, done.shape, start.shape)
        print(s.shape, kv.shape, z.shape, key.shape)
        breakpoint()
        s, z, _, _ = lax.associative_scan(self.binary_op, (kv, key, start, done))
        # s, _, _ = lax.associative_scan(self.binary_s_op, (kv, start, done))
        # z, _, _ = lax.associative_scan(self.binary_z_op, (key, start, done))

        numer = jnp.einsum("ti, tij -> tj", query, s)
        denom = jnp.einsum("ti, tj -> t", z, query).clip(1e-5)

        output = numer / denom  # jnp.clip(denom, 1e-5)
        output = eqx.filter_vmap(self.ln)(
            eqx.filter_vmap(self.mlp)(output) + eqx.filter_vmap(self.skip)(x)
        )

        return output, [s[-2:-1], z[-2:-1]]


class LTQNetwork(eqx.Module):
    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: List[eqx.Module]
    post: eqx.Module
    name: str = "LT"

    def __init__(self, obs_shape, act_shape, config, key):
        self.config = config
        self.input_size = obs_shape
        self.output_size = act_shape
        keys = random.split(key, 5)
        self.pre = nn.Sequential(
            [nn.Linear(obs_shape, config["mlp_size"], key=keys[0]), mish]
        )
        self.memory = [
            LinearAttention(config["mlp_size"], config["recurrent_size"], keys[1]),
            LinearAttention(
                config["recurrent_size"], config["recurrent_size"], keys[2]
            ),
        ]
        self.post = nn.Sequential(
            [
                nn.Linear(self.config["recurrent_size"], self.output_size, key=keys[3]),
            ]
        )

    @eqx.filter_jit
    def __call__(self, x, state, start, done):
        x = vmap(self.pre)(x)

        states = []
        for i, mod in enumerate(self.memory):
            x, (s_i, z_i) = mod(x, state[i], start, done)
            states.append((s_i, z_i))

        y = vmap(self.post)(x)
        return y, states

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return [
            (
                jnp.zeros(
                    (
                        *shape,
                        1,
                        self.config["recurrent_size"],
                        self.config["recurrent_size"],
                    ),
                    dtype=jnp.float32,
                ),
                jnp.zeros(
                    (*shape, 1, self.config["recurrent_size"]), dtype=jnp.float32
                ),
            )
            for _ in range(2)
        ]
