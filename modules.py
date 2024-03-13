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

def complex_symlog(x, key=None):
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x)) * jnp.exp(1j * jnp.angle(x))

def linear_softplus(x, key=None):
    parabolic_constant = jnp.arcsinh(1) + jnp.sqrt(2)
    return jnp.log(1 + jnp.exp(x * parabolic_constant))

def soft_relglu(x, key=None):
    return linear_softplus(jax.nn.glu(x))

def mish(x, key=None):
    return x * jnp.tanh(jax.nn.softplus(x))

def leaky_relu(x, key=None):
    return jax.nn.leaky_relu(x, negative_slope=0.01)

def gelu(x, key=None):
    return jax.nn.gelu(x)

def elu(x, key=None):
    return jax.nn.elu(x)

def smooth_leaky_relu(x, key=None):
    b = 0.05
    return (x < 0) * (jnp.exp((1 - b) * x) + b * x - 1.0) + (x >= 0) * x
    

def gaussian(x, key=None):
    return jnp.exp(-x ** 2)


class RandomSequential(nn.Sequential):
    def __call__(self, x, key=None):
        return super().__call__(x, key=key)

class Block(eqx.Module):
    net: eqx.Module
    def __init__(self, input_size, output_size, dropout, key):
        if dropout == 0.0:
            self.net = RandomSequential([
                nn.Linear(input_size, output_size, key=key), 
                nn.LayerNorm(output_size, use_weight=False, use_bias=False),
                leaky_relu,
            ])
        else:
            self.net = RandomSequential([
                nn.Linear(input_size, output_size, key=key), 
                nn.LayerNorm(output_size, use_weight=False, use_bias=False),
                nn.Dropout(dropout),
                leaky_relu,
            ])

    def __call__(self, x, key=None):
        return self.net(x, key=key)

class QHead(eqx.Module):
    post0: eqx.Module
    post1: eqx.Module
    value: nn.Linear
    advantage: nn.Linear

    def __init__(self, input_size, hidden_size, output_size, dropout, key):
        keys = random.split(key, 3)

        self.post0 = eqx.filter_vmap(Block(input_size, hidden_size, dropout, keys[0]))
        self.post1 = eqx.filter_vmap(Block(hidden_size, hidden_size, dropout, keys[1]))
        self.value = eqx.filter_vmap(final_linear(keys[2], input_size, 1, scale=0.01))
        self.advantage = eqx.filter_vmap(final_linear(keys[3], input_size, output_size, scale=0.01))

    def __call__(self, x, key):
        T = x.shape[0]
        net_keys = random.split(key, 2 * T)
        x = self.post0(x, net_keys[:T])
        x = self.post1(x, net_keys[T:2*T])
        V = self.value(x) 
        A = self.advantage(x)
        # Dueling DQN
        return V + (A - A.mean(axis=-1, keepdims=True))

        

class AtariCNN(eqx.Module):
    c0: nn.Conv2d
    c1: nn.Conv2d
    c2: nn.Conv2d
    ln0: nn.LayerNorm
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm
    linear: nn.Linear

    def __init__(self, output_size, key):
        keys = random.split(key, 4)
        self.c0 = nn.Conv2d(1, 32, kernel_size=8, stride=4, key=keys[0])
        self.ln0 = nn.LayerNorm((32, 20, 20), use_weight=False, use_bias=False)
        self.c1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, key=keys[1])
        self.ln1 = nn.LayerNorm((64, 9, 9), use_weight=False, use_bias=False)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, key=keys[2])
        self.ln2 = nn.LayerNorm((64, 7, 7), use_weight=False, use_bias=False)
        self.linear = nn.Linear(7 * 7 * 64, output_size, key=keys[3])

    def __call__(self, x, keys=None):
        x = x.reshape(-1, 84, 84)
        x = leaky_relu(self.ln0(self.c0(x)))
        x = leaky_relu(self.ln1(self.c1(x)))
        x = leaky_relu(self.ln2(self.c2(x)))
        # x = leaky_relu(self.c0(x))
        # x = leaky_relu(self.c1(x))
        # x = leaky_relu(self.c2(x))
        x = self.linear(x.flatten())
        return x


class RecurrentQNetwork(eqx.Module):
    """The core model used in experiments"""
    input_size: int
    output_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: eqx.Module
    q: eqx.Module

    def __init__(self, obs_shape, act_shape, memory_module, config, key):
        self.config = config
        self.output_size = act_shape
        keys = random.split(key, 4)
        if config.get("atari_cnn"):
            self.input_size = (84, 84)
            self.pre = eqx.filter_vmap(AtariCNN(config["mlp_size"], keys[1]))
        else:
            [self.input_size] = obs_shape
            self.pre = eqx.filter_vmap(Block(self.input_size, config["mlp_size"], 0, keys[1]))
        self.memory = memory_module

        ensemble_keys = random.split(keys[0], config["ensemble_size"])

        @eqx.filter_vmap
        def make_heads(key):
            return QHead(config["recurrent_size"], config["mlp_size"], act_shape, config["dropout"], key)
                    
        self.q = make_heads(ensemble_keys)


    def __call__(self, x, state, start, done, key):
        T = x.shape[0]
        net_keys = random.split(key, T + 1)
        x = self.pre(x, net_keys[:T])
        y, final_state = self.memory(x=x, state=state, start=start, next_done=done, key=key)

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
        def ensemble(model, x, key):
            return model(x, key=key)

            
        q = ensemble(self.q, y, net_keys[-1])
        return q, final_state

    def evaluate(self, x, state, start, done, key):
        q = self(x, state, start, done, key)
        return jnp.median(q, axis=0)

    def initial_state(self, shape=tuple()):
        return self.memory.initial_state(shape)


def mean_noise(network):
    leaves = jax.tree_leaves(network, is_leaf=lambda x: isinstance(x, NoisyLinear))
    nelem = sum(leaf.sigma_weight.size for leaf in leaves if isinstance(leaf, NoisyLinear))
    sum_ = sum(jnp.sum(leaf.sigma_weight) for leaf in leaves if isinstance(leaf, NoisyLinear))
    result = sum_ / nelem
    return result


class Lambda(eqx.Module):
    f: Callable

    def __init__(self, f):
        self.f = f

    def __call__(self, x, key=None):
        return self.f(x)


def ortho_init(key, linear, scale):
    init = jax.nn.initializers.orthogonal(scale=scale)
    linear = eqx.tree_at(lambda l: l.weight, linear, init(key, linear.weight.shape))
    linear = eqx.tree_at(lambda l: l.bias, linear, jnp.zeros_like(linear.bias))
    return linear

def default_init(key, linear, scale=1.0, zero_bias=False, fixed_bias=None):
    lim = math.sqrt(scale / linear.in_features)
    linear = eqx.tree_at(lambda l: l.weight, linear, jax.random.uniform(key, linear.weight.shape, minval=-lim, maxval=lim))
    if zero_bias:
        linear = eqx.tree_at(lambda l: l.bias, linear, jnp.zeros_like(linear.bias))
    elif fixed_bias is not None:
        linear = eqx.tree_at(lambda l: l.bias, linear, jnp.full_like(linear.bias, fixed_bias))
    return linear


def ortho_linear(key, input_size, output_size, scale=2 ** 0.5):
    return ortho_init(key, eqx.nn.Linear(input_size, output_size, key=key), scale=scale)

def final_linear(key, input_size, output_size, scale=0.01):
    #linear = ortho_linear(key, input_size, output_size, scale=scale)
    linear = nn.Linear(input_size, output_size, key=key)
    linear = default_init(key, linear, scale=scale)
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


def anneal(epsilon_start, epsilon_end, progress):
    return epsilon_start + (epsilon_end - epsilon_start) * progress


def make_random_policy(env):
    def random_policy(q_network, x, state, *args, **kwargs):
        return env.action_space.sample(), state


def greedy_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end
):
    q_values, state = q_network(jnp.expand_dims(x, 0), state, start, done, key=key)
    action = jnp.argmax(q_values)
    return action, state

def ensemble_greedy_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end
):
    q_values, state = q_network(jnp.expand_dims(x, 0), state, start, done, key=key)
    #action = jnp.argmax(q_values.min(0))
    #action = jnp.argmax(jnp.median(q_values, axis=0))
    ensemble_reduced = jax.random.choice(key, q_values, tuple(), axis=0)
    action = jnp.argmax(ensemble_reduced)
    return action, state

def boltzmann_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end
):
    _, q_key, s_key  = random.split(key, 3)
    q_values, state = q_network(jnp.expand_dims(x, 0), state, start, done, key=q_key)
    ensemble_reduced = jax.random.choice(key, q_values, tuple(), axis=0)
    temp = anneal(epsilon_start, epsilon_end, progress)
    divisor = 1e-6 + ensemble_reduced.std() * temp
    action = jax.random.categorical(s_key, ensemble_reduced / divisor, axis=-1).squeeze(-1)
    return action, state

def sum_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end
):
    e_key, q_key, s_key  = random.split(key, 3)
    q_values, state = q_network(jnp.expand_dims(x, 0), state, start, done, key=q_key)
    #action = jax.random.categorical(s_key, q_values / temp, axis=-1).squeeze(-1)
    ensemble_reduced_q = jax.random.choice(e_key, q_values, tuple(), axis=0)
    probs = ensemble_reduced_q / jnp.sum(ensemble_reduced_q, keepdims=True)
    action = jax.random.choice(s_key, jnp.arange(q_values.shape[-1]), (1,), p=probs, axis=-1)
    return action, state

def epsilon_greedy_policy(
    q_network, x, state, start, done, key, progress, epsilon_start, epsilon_end, base_policy=greedy_policy
):
    _, p_key, r_key, s_key = random.split(key, 4)
    action, state = base_policy(
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
    params = eqx.filter(network, eqx.is_inexact_array)
    _, static = eqx.partition(target, eqx.is_inexact_array)
    target = eqx.combine(static, params)
    return target


def soft_update(network, target, tau):
    def polyak(param, target_param):
        return target_param * (1 - tau) + param * tau

    params, _ = eqx.partition(network, eqx.is_inexact_array)
    target_params, static = eqx.partition(target, eqx.is_inexact_array)
    updated_params = jax.tree_map(polyak, params, target_params)
    target = eqx.combine(static, updated_params)
    target = eqx.tree_inference(target, True)
    return target

def shrink_perturb_soft_update(network, target, random, tau1, tau2=1/400):
    def polyak(param, target_param, rand_param):
        return target_param * (1 - tau1 - tau2) + param * tau1 + random * tau2

    params, _ = eqx.partition(network, eqx.is_inexact_array)
    target_params, static = eqx.partition(target, eqx.is_inexact_array)
    updated_params = jax.tree_map(polyak, params, target_params)
    target = eqx.combine(static, updated_params)
    return target

