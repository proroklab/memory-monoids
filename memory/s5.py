from typing import Any, Callable, Dict, List, Tuple
import equinox as eqx
from equinox import nn
import jax
from jax import random, vmap, lax
import jax.numpy as jnp
import jax.numpy as np
from memory.module import MemoryModule
import jax.nn as jnn
from utils import expand_right
from jax.numpy.linalg import eigh
from jax.nn.initializers import lecun_normal, normal

# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
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
    A, b = carry

    A = jnp.logical_not(start) * A + start * jnp.ones_like(A)
    b = jnp.logical_not(start) * b

    out = binary_operator((A, b), incoming)
    start_out = jnp.logical_or(start, prev_start)
    return (start_out, *out)


def apply_ssm(Lambda_bar, B_bar, C_tilde, hidden, input_sequence, start, conj_sym, bidirectional=False):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            start      (bool): input sequence of features                (L,)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence.astype(jnp.complex64))

    Lambda_elements = jnp.concatenate([
        jnp.ones((1, Lambda_bar.shape[0])),
        Lambda_elements,
    ])

    Bu_elements = jnp.concatenate([
        hidden,
        Bu_elements,
    ])

    start = start.reshape([-1, 1])
    start = jnp.concatenate([
        jnp.zeros_like(start[:1]),
        start,
    ], axis=0)

    _, _, xs = jax.lax.associative_scan(wrapped_associative_update, (start, Lambda_elements, Bu_elements))
    xs = xs[1:]

    if conj_sym:
        return xs[np.newaxis, -1], jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return xs[np.newaxis, -1], jax.vmap(lambda x: (C_tilde @ x).real)(xs)


def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar

def init_CV(init_fun, rng, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(key, shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)

class S5SSM(eqx.Module):
    # ACTUAL PARAMS
    Lambda_re: jax.Array
    Lambda_im: jax.Array
    B: jax.Array
    C: jax.Array
    log_step: jax.Array
    D: jax.Array

    H: int
    P: int
    C_init: str = "lecun_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def __init__(self, d_model, ssm_size, blocks, key):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """
        keys = jax.random.split(key, 4)
        self.H = d_model

        block_size = int(ssm_size / blocks)
        Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2 # WHY
        ssm_size = ssm_size // 2

        self.P = ssm_size
        Lambda = Lambda[:block_size]

        V = V[:, :block_size]
        Vinv = V.conj().T
        self.Lambda_re = Lambda.real
        self.Lambda_im = Lambda.imag

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        # self.Lambda_re = Lambda_re_init
        # self.Lambda_im = Lambda_im_init

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = init_VinvB(B_init, keys[0], B_shape, Vinv)

        # Initialize state to output (C) matrix
        # if self.C_init in ["trunc_standard_normal"]:
        #     C_init = trunc_standard_normal
        #     C_shape = (self.H, local_P, 2)
        if self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            self.C = C_init(keys[1], (self.H, self.P, 2))
            # self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            self.C = init_CV(C_init, keys[1], C_shape, V)
            # self.C_tilde = C[..., 0] + 1j * C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = normal(stddev=1.0)(keys[2], (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = init_log_steps(keys[3], (self.P, self.dt_min, self.dt_max))
        # self.log_step = self.param("log_step",
        #                            init_log_steps,
        #                            (self.P, self.dt_min, self.dt_max))

        # Discretize
        # self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)

    def __call__(self, hidden, input_sequence, start):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
             resets (bool): input sequence (L,)
        Returns:
            output sequence (float32): (L, H)
        """
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * np.exp(self.log_step[:, 0])
        Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        hidden, ys = apply_ssm(Lambda_bar,
                       B_bar,
                       C_tilde,
                       hidden,
                       input_sequence,
                       start,
                       self.conj_sym,
                       self.bidirectional)
        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return hidden, ys + Du


class S5Layer(eqx.Module):
    ssm: eqx.Module
    ln: eqx.Module
    out1: eqx.Module
    out2: eqx.Module
    d_model: int

    def __init__(self, d_model, ssm_size, blocks, key):
        keys = random.split(key, 3)
        # self.ssm = eqx.filter_vmap(S5SSM(step_rescale=1.0))
        self.d_model = d_model
        self.ssm = S5SSM(
            d_model, ssm_size, blocks, keys[0]
        )
        self.ln = eqx.filter_vmap(nn.LayerNorm(self.d_model))
        self.out1 = eqx.filter_vmap(nn.Linear(self.d_model, self.d_model, key=keys[1]))
        self.out2 = eqx.filter_vmap(nn.Linear(self.d_model, self.d_model, key=keys[2]))
    
    def __call__(self, state, x, start):
        skip = x
        # if self.prenorm:
        x = self.ln(x)
        hidden, x = self.ssm(state, x, start)
        x = jax.nn.gelu(x)
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
        return hidden, skip + x

class StackedS5(MemoryModule):
    layers: List[eqx.Module]
    encoder: eqx.Module
    d_model: int
    n_layers: int
    ssm_size: int
    name: str = "StackedS5"

    def __init__(self, input_size, n_layers, d_model, ssm_size, blocks, key):
        keys = random.split(key, n_layers+1)
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.ssm_size = ssm_size

        self.encoder = nn.Linear(input_size, self.d_model, key=keys[0])
        self.layers = [
            S5Layer(d_model, ssm_size, blocks, keys[i+1]) for i in range(n_layers)
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
                (1, *shape, self.ssm_size // 2), dtype=jnp.complex64
            ) for _ in range(self.n_layers)
        ]