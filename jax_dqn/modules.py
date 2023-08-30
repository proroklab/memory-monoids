from typing import Any, Callable, Dict
import jax
import equinox as eqx
from jax import random
import jax.numpy as jnp

@jax.jit
def mish(x, key=None):
    return x * jnp.tanh(jax.nn.softplus(x))

@eqx.filter_jit
class Lambda(eqx.Module):
    f: Callable
    def __init__(self, f):
        self.f = f

    def __call__(self, x, key=None):
        return self.f(x)

class FinalLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, input_size, output_size, key):
        self.weight = random.normal(key, (input_size, output_size)) * 0.001
        self.bias = jnp.zeros(output_size)

    def __call__(self, x, key=None):
        return x @ self.weight + self.bias



@jax.jit
def anneal(epsilon_start, epsilon_end, progress):
    return epsilon_start + (epsilon_end - epsilon_start) * progress


def make_random_policy(env):
    def random_policy(q_network, x, state, *args, **kwargs):
        return env.action_space.sample(), state


@eqx.filter_jit
def greedy_policy(q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end):
    start = jnp.array([start])
    done = jnp.array([done])
    q_values, state = q_network(jnp.expand_dims(x, 0), state, start, done)
    action = jnp.argmax(q_values)
    return action, state

@eqx.filter_jit
def epsilon_greedy_policy(q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end):
    _, *keys = random.split(key, 3)
    action, state = greedy_policy(q_network, x, state, start, done, None, None, None, None)
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

def soft_update(network, target, tau=1/30):
    def polyak(param, target_param):
        return target_param * (1 - tau) + param * tau
    params, _ = eqx.partition(network, eqx.is_inexact_array)
    target_params, static = eqx.partition(target, eqx.is_inexact_array)
    updated_params = jax.tree_map(polyak, params, target_params)
    target = eqx.combine(static, updated_params)
    return target
