from typing import Callable
import jax
import equinox as eqx
from equinox import nn
from jax import random, vmap
import jax.numpy as jnp

@jax.jit
def mish(x, key=None):
    return x * jnp.tanh(jax.nn.softplus(x))

class FinalLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, input_size, output_size, key):
        self.weight = random.normal(key, (input_size, output_size)) * 0.001
        self.bias = jnp.zeros(output_size)

    def __call__(self, x, key=None):
        return x @ self.weight + self.bias


class QNetwork(eqx.Module):
    input_size: int
    output_size: int
    state_size: int
    hidden_size: int
    pre: eqx.Module
    memory: eqx.Module
    post: eqx.Module

    def __init__(self, obs_shape, mlp_size, recurrent_size, act_shape, key):
        self.input_size = obs_shape
        self.output_size = act_shape
        self.state_size = recurrent_size
        self.hidden_size = mlp_size
        keys = random.split(key, 5)
        self.pre = nn.Sequential([nn.Linear(obs_shape, mlp_size, key=keys[0]), mish])
        self.memory = nn.GRUCell(mlp_size, self.state_size, key=keys[1])
        self.post = nn.Sequential(
            [
                nn.Linear(self.state_size, self.hidden_size, key=keys[2]),
                mish,
                nn.Linear(self.hidden_size, self.hidden_size, key=keys[3]),
                mish,
                FinalLinear(self.hidden_size, self.output_size, key=keys[4])
            ]
        )

    @eqx.filter_jit
    def __call__(self, x, state):
        x = vmap(self.pre)(x)

        def scan_fn(state, input):
            state = self.memory(input, state)
            return state, state

        final_state, state = jax.lax.scan(scan_fn, state, x)
        y = vmap(self.post)(state)
        return y, final_state

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return jnp.zeros((*shape, self.state_size), dtype=jnp.float32)


@jax.jit
def anneal(epsilon_start, epsilon_end, progress):
    return epsilon_start + (epsilon_end - epsilon_start) * progress


def make_random_policy(env):
    def random_policy(q_network, x, state, *args, **kwargs):
        return env.action_space.sample(), state


@eqx.filter_jit
def greedy_policy(q_network, x, state, *args, **kwargs):
    q_values, state = q_network(jnp.expand_dims(x, 0), state)
    action = jnp.argmax(q_values)
    return action, state

@eqx.filter_jit
def epsilon_greedy_policy(q_network, x, state, key, progress, epsilon_start, epsilon_end):
    _, *keys = random.split(key, 3)
    action, state = greedy_policy(q_network, x, state)
    random_action = random.randint(keys[0], (), 0, q_network.output_size)
    action = jax.lax.cond(
        random.uniform(keys[1]) < anneal(epsilon_start, epsilon_end, progress),
        lambda: random_action,
        lambda: action,
    )
    return action, state


def hard_update(network, target):
    params, _ = eqx.partition(network, eqx.is_inexact_array)
    _, static = eqx.partition(target, eqx.is_inexact_array)
    target = eqx.combine(static, params)
    return target