import math
from typing import Any, Callable, Dict
import jax
import equinox as eqx
from equinox import nn
from jax import random
import jax.numpy as jnp


def symlog(x, key=None):
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

def linsymlog(x, key=None):
    return jnp.sign(x) * (1 + jnp.abs(x)) + jnp.tanh(x)

def softsymlog(x, key=None):
    return jnp.tanh(x) * (1 + jnp.log(1 + jnp.abs(x)))

def leakytanh(x, key=None):
    return jnp.tanh(x) + 0.1 * x

@jax.jit
def mish(x, key=None):
    return x * jnp.tanh(jax.nn.softplus(x))

# @jax.jit
# def mish(x, key=None):
#     #return x * jnp.pi / (jnp.pi + jnp.exp(-x)) 
#     #return x / (1 + jnp.exp(-x)) 
#     #return x / (1 + jnp.abs(x))
#     #return x - jnp.tanh(x)
#     return jnp.sign(x) * jnp.log(1 + jnp.exp(jnp.abs(x)))


class RandomSequential(nn.Sequential):
    def __call__(self, x, key=None):
        return super().__call__(x, key=key)

class Block(eqx.Module):
    net: eqx.Module
    def __init__(self, input_size, output_size, key):
        self.net = RandomSequential([
            ortho_linear(key, input_size, output_size), 
            #nn.LayerNorm(None, use_bias=False, use_weight=False,),
            mish,
        ])

    def __call__(self, x, key=None):
        return self.net(x, key=key)


class RecurrentQNetwork(eqx.Module):
    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: eqx.Module
    post0: eqx.Module
    post1: eqx.Module
    value: eqx.Module
    advantage: eqx.Module
    scale: eqx.Module

    def __init__(self, obs_shape, act_shape, memory_module, config, key):
        self.config = config
        self.input_size = obs_shape
        self.output_size = act_shape
        keys = random.split(key, 7)
        self.pre = eqx.filter_vmap(Block(obs_shape, config["mlp_size"], keys[1]))
        self.memory = memory_module
        self.post0 = eqx.filter_vmap(Block(config["recurrent_size"], config["mlp_size"], keys[2]))
        self.post1 = eqx.filter_vmap(Block(config["mlp_size"], config["mlp_size"], keys[3]))

        value = final_linear(keys[4], self.config["mlp_size"], 1, scale=0.01)
        self.value = eqx.filter_vmap(value)
        advantage = ortho_linear(keys[5], self.config["mlp_size"], self.output_size)
        self.advantage = eqx.filter_vmap(advantage)
        scale = final_linear(keys[6], self.config["mlp_size"], 1, scale=0.01)
        self.scale = eqx.filter_vmap(scale)

    @eqx.filter_jit
    def __call__(self, x, state, start, done, key):
        T = x.shape[0]
        net_keys = random.split(key, 3 * T)
        x = self.pre(x, net_keys[:T])
        y, final_state = self.memory(x=x, state=state, start=start, next_done=done, key=key)
        y = self.post0(y, net_keys[T:2*T])
        y = self.post1(y, net_keys[2*T:])

        value = self.value(y)
        A = self.advantage(y)
        scale = self.scale(y)

        A_normed = A / (1e-6 + jnp.linalg.norm(A, axis=-1, keepdims=True))
        A_normed = A / A.max(axis=-1, keepdims=True) 
        advantage = A_normed - jnp.mean(A_normed, axis=-1, keepdims=True)
        # TODO: Only use target network for advantage branch
        # Let value/scale increase as needed
        q = value + scale * advantage
        return q, final_state

    @eqx.filter_jit
    def initial_state(self, shape=tuple()):
        return self.memory.initial_state(shape)


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
    normalize: bool

    def __init__(self, input_size, output_size, init_std=0.017, normalize=False, *, key, inference=False):
        super().__init__(input_size, output_size, key=key)
        self.sigma_bias = jnp.ones(self.bias.shape) * init_std 
        self.sigma_weight = jnp.ones(self.weight.shape) * init_std 
        self.inference = inference
        self.normalize = normalize

    def get_noise(self):
        return self.sigma_weight

    def __call__(self, x, key=None):
        if self.inference:
            weight = self.weight
            bias = self.bias
        else:
            # _, bkey, wkey = random.split(key, 3)
            # weight = self.weight + self.sigma_weight * random.normal(wkey, self.weight.shape)
            # bias = self.bias + self.sigma_bias * random.normal(bkey, self.bias.shape)
            _, bkey, wkey = random.split(key, 3)
            bnoise = random.normal(bkey, self.bias.shape)
            wnoise = jnp.outer(bnoise, random.normal(wkey, self.weight.shape[1:]))
            wnoise = jnp.sign(wnoise) * jnp.sqrt(jnp.abs(wnoise))
            if self.normalize:
                sigma_bias = self.sigma_bias / (1e-6 + jnp.linalg.norm(self.sigma_bias, keepdims=True))
                sigma_weight = self.sigma_weight / (1e-6 + jnp.linalg.norm(self.sigma_weight, keepdims=True))
            else:
                sigma_bias = self.sigma_bias
                sigma_weight = self.sigma_weight

            bias = self.bias + sigma_bias * bnoise
            weight = self.weight + sigma_weight * wnoise

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


@eqx.filter_jit
def hard_update(network, target):
    params = eqx.filter(network, eqx.is_inexact_array)
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