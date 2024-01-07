# https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py
from functools import partial
import jax
import jax.numpy as jnp
# from flax import linen as nn
from memory.module import MemoryModule
from modules import Lambda, leaky_relu
import equinox as eqx
from equinox import nn
from typing import List, Tuple

parallel_scan = jax.lax.associative_scan

# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

def wrapped_associative_update(carry: jax.Array, incoming: jax.Array) -> Tuple[jax.Array, ...]:
    """The reset-wrapped form of the associative update. 

    You might need to override this
    if you use variables in associative_update that are not from initial_state. 
    This is equivalent to the h function in the paper:
    b x H -> b x H
    """
    prev_start, *carry = carry
    start, *incoming = incoming
    # Reset all elements in the carry if we are starting a new episode
    s, z = carry

    s = jnp.logical_not(start) * s
    z = jnp.logical_not(start) * z

    out = binary_operator_diag((s, z), incoming)
    start_out = jnp.logical_or(start, prev_start)
    return (start_out, *out)


def matrix_init(key, shape, dtype=jnp.float_, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float_):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float_):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(eqx.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    theta_log: jax.Array
    nu_log: jax.Array
    gamma_log: jax.Array
    B_re: jax.Array
    B_im: jax.Array
    C_re: jax.Array
    C_im: jax.Array
    D: jax.Array

    d_model: int  # input and output dimensions
    d_hidden: int  # hidden state dimension
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda

    def __init__(self, d_model, d_hidden, key):
        keys = jax.random.split(key, 8)
        self.d_hidden = d_hidden
        self.d_model = d_model

        self.theta_log = theta_init(keys[0], (self.d_hidden,), self.max_phase)
        self.nu_log = nu_init(keys[1], (self.d_hidden,), self.r_min, self.r_max)
        self.gamma_log = gamma_log_init(keys[2], (self.nu_log, self.theta_log))
        self.B_re = matrix_init(keys[3], (self.d_hidden, self.d_model), normalization=jnp.sqrt(2 * self.d_model))
        self.B_im = matrix_init(keys[4], (self.d_hidden, self.d_model), normalization=jnp.sqrt(2 * self.d_model))
        self.C_re = matrix_init(keys[5], (self.d_model, self.d_hidden), normalization=jnp.sqrt(self.d_hidden))
        self.C_im = matrix_init(keys[6], (self.d_model, self.d_hidden), normalization=jnp.sqrt(self.d_hidden))
        self.D = matrix_init(keys[7], (self.d_model,))

    def __call__(self, state, x, start):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)
        C = self.C_re + 1j * self.C_im

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)

        Lambda_elements = jnp.concatenate([
            jnp.ones((1, diag_lambda.shape[0])),
            Lambda_elements,
        ])

        Bu_elements = jnp.concatenate([
            state,
            Bu_elements,
        ])

        start = start.reshape([-1, 1])
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)

        # Compute hidden states
        _, _, xs = parallel_scan(wrapped_associative_update, (start, Lambda_elements, Bu_elements))
        xs = xs[1:]

        # Use them to compute the output of the module
        outputs = jax.vmap(lambda x, u: (C @ x).real + self.D * u)(xs, x)

        return xs[None, -1], outputs


class SequenceLayer(eqx.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    lru: LRU  # lru module
    d_model: int  # model size
    d_hidden: int # hidden size
    out1: eqx.Module  # first output linear layer
    out2: eqx.Module  # second output linear layer
    normalization: eqx.Module  # layer norm

    def __init__(self, d_model, d_hidden, key):
        """Initializes the ssm, layer norm and dropout"""
        keys = jax.random.split(key, 3)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.lru = LRU(self.d_model, d_hidden, key=keys[0])
        self.out1 = eqx.filter_vmap(nn.Linear(self.d_model, self.d_model, key=keys[1]))
        self.out2 = eqx.filter_vmap(nn.Linear(self.d_model, self.d_model, key=keys[2]))
        self.normalization = eqx.filter_vmap(nn.LayerNorm(self.d_model))

    def __call__(self, state, x, start):
        skip = x
        x = self.normalization(x)  # pre normalization
        state, x = self.lru(state, x, start)  # call LRU
        x = jax.nn.gelu(x)
        o1 = self.out1(x)
        x = o1 * jax.nn.sigmoid(self.out2(x))  # GLU
        return state, skip + x  # skip connection


class StackedLRU(MemoryModule):
    """Encoder containing several SequenceLayer"""

    layers: List[SequenceLayer]
    encoder: eqx.Module
    d_model: int
    d_hidden: int
    n_layers: int
    name: str = "StackedLRU"

    def __init__(self, input_size, d_model, d_hidden, n_layers, key):
        keys = jax.random.split(key, 7)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        # self.encoder = nn.Dense(self.d_model)
        self.encoder = nn.Linear(input_size, self.d_model, key=keys[0])
        self.layers = [
            SequenceLayer(
                d_model=self.d_model,
                d_hidden=self.d_hidden,
                key=keys[i+1]
            )
            for i in range(self.n_layers)
        ]

    def __call__(self, x, state, start, next_done, key=None):
        new_states = []
        for i, layer in enumerate(self.layers):
            new_s, x = layer(state[i], x, start)
            new_states.append(new_s)
    
        return x, new_states

    def initial_state(self, shape=tuple()):
        return [
            jnp.zeros(
                (1, *shape, self.d_hidden), dtype=jnp.complex64
            ) for _ in range(self.n_layers)
        ]