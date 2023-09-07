from functools import partial
import math
from typing import Any, Callable, Dict
import jax
import equinox as eqx
from jax import random
import jax.numpy as jnp


@jax.jit
def mish(x, key=None):
    return x * jnp.tanh(jax.nn.softplus(x))

def mean_noise(network):
    leaves = jax.tree_leaves(network, is_leaf=lambda x: isinstance(x, NoisyLinear))
    nelem = sum(leaf.sigma_weight.size for leaf in leaves if isinstance(leaf, NoisyLinear))
    sum_ = sum(jnp.sum(leaf.sigma_weight) for leaf in leaves if isinstance(leaf, NoisyLinear))
    result = sum_ / nelem
    return result


@eqx.filter_jit
class Lambda(eqx.Module):
    f: Callable

    def __init__(self, f):
        self.f = f

    def __call__(self, x, key=None):
        return self.f(x)


def ortho_init(key, linear, scale):
    init = jax.nn.initializers.orthogonal(scale=scale)
    linear = eqx.tree_at(lambda l: l.weight, linear, init(key, linear.weight.shape))
    return linear

def ortho_linear(key, input_size, output_size, scale=1.0):
    return ortho_init(key, eqx.nn.Linear(input_size, output_size, key=key), scale=scale)

def final_linear(key, input_size, output_size, scale=0.01):
    linear = ortho_linear(key, input_size, output_size, scale=scale)
    linear = eqx.tree_at(lambda l: l.bias, linear, linear.bias * 0.0)
    return linear


class NoisyLinear(eqx.nn.Linear):
    sigma_weight: jax.Array
    sigma_bias: jax.Array
    inference: bool

    def __init__(self, input_size, output_size, init_std=0.017, *, key, inference=False):
        super().__init__(input_size, output_size, key=key)
        self.sigma_bias = jnp.ones(self.bias.shape) * init_std 
        self.sigma_weight = jnp.ones(self.weight.shape) * init_std 
        self.inference = inference

    def get_noise(self):
        return self.sigma_weight

    def __call__(self, x, key=None):
        if self.inference:
            weight = self.weight
            bias = self.bias
        else:
            _, bkey, wkey = random.split(key, 3)
            weight = self.weight + self.sigma_weight * random.normal(wkey, self.weight.shape)
            bias = self.bias + self.sigma_bias * random.normal(bkey, self.bias.shape)

        return weight @ x + bias

        # lin = self.weight @ x + self.bias
        # if self.inference:
        #     return lin
        # bias_noise = self.sigma_bias * random.normal(bkey, self.bias.shape)
        # weight_noise = self.sigma_weight * random.normal(wkey, self.weight.shape) @ x
        # return lin + bias_noise + weight_noise


#@jax.jit
def anneal(epsilon_start, epsilon_end, progress):
    return epsilon_start + (epsilon_end - epsilon_start) * progress


def make_random_policy(env):
    def random_policy(q_network, x, state, *args, **kwargs):
        return env.action_space.sample(), state


@eqx.filter_jit
def greedy_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end
):
    q_values, state = q_network(jnp.expand_dims(x, 0), state, start, done, key=key)
    action = jnp.argmax(q_values)
    return action, state

@eqx.filter_jit
def boltzmann_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end
):
    _, q_key, s_key  = random.split(key, 3)
    temp = anneal(epsilon_start, epsilon_end, progress)
    q_values, state = q_network(jnp.expand_dims(x, 0), state, start, done, key=q_key)
    action = jax.random.categorical(s_key, q_values / temp, axis=-1).squeeze(-1)
    return action, state

@eqx.filter_jit
def epsilon_greedy_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end
):
    _, p_key, r_key, s_key = random.split(key, 4)
    action, state = greedy_policy(
        q_network, x, state, start, done, key=p_key, progress=None, epsilon_start=None, epsilon_end=None
    )
    random_action = random.randint(r_key, (), 0, q_network.output_size)
    do_random = random.uniform(s_key) < anneal(epsilon_start, epsilon_end, progress)
    action = jax.lax.select(
        do_random,
        random_action,
        action,
    )
    return action, state


def hard_update(network, target):
    params, _ = eqx.partition(network, eqx.is_inexact_array)
    _, static = eqx.partition(target, eqx.is_inexact_array)
    target = eqx.combine(static, params)
    return target


@eqx.filter_jit
def soft_update(network, target, tau):
    def polyak(param, target_param):
        return target_param * (1 - tau) + param * tau

    params, _ = eqx.partition(network, eqx.is_inexact_array)
    target_params, static = eqx.partition(target, eqx.is_inexact_array)
    updated_params = jax.tree_map(polyak, params, target_params)
    target = eqx.combine(static, updated_params)
    return target