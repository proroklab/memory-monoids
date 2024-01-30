from functools import partial
from typing import List, Tuple
import jax
from jax import numpy as jnp
import equinox as eqx

from modules import softsymlog, soft_update


def huber(x):
    return jax.lax.select(
        jnp.abs(x) < 1.0,
        0.5 * x ** 2,
        jnp.abs(x) - 0.5 
    )

def gaussian_nll(x, logvar):
    return 0.5 * (
        logvar + x ** 2 / (1e-6 + jnp.exp(logvar))
        
        + x ** 2
        / jnp.max(logvar, 1e-6)
    )

def cauchy(x):
    return jnp.log(1 + x ** 2)

def cauchy_abs(x):
    return jnp.log(1 + jnp.abs(x))

def masked_mean(x, mask):
    return jnp.sum(x * mask) / mask.sum()

def nan_breakpoint(x):
    def true_fn(x):
        pass
    def false_fn(x):
        jax.debug.breakpoint()
    jax.lax.cond(jnp.isfinite(x).all(), true_fn, false_fn, x)

def tempered_softmax(x, temp=10, key=None):
    return jax.nn.softmax(x * temp)

def segment_ddqn_loss(q_network, q_target, segment, gamma, key):
    # Double DQN
    B, T = segment["next_reward"].shape
    initial_state = q_network.initial_state()
    q_values, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0, None))(
        segment["observation"], initial_state, segment["start"], segment["next_terminated"], key
    )
    batch_index = jnp.repeat(jnp.arange(B), T)
    time_index = jnp.tile(jnp.arange(T), B)
    # B, ensemble, T, action
    selected_q = q_values.squeeze(1)[
        batch_index, time_index, segment["action"].reshape(-1)
    ].reshape(segment["action"].shape)

    next_q_action_idx, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0, None))(
        segment["next_observation"], initial_state, segment["start"], segment["next_terminated"], key
    )
    next_q, _ = jax.lax.stop_gradient(eqx.filter_vmap(q_target, in_axes=(0, None, 0, 0, None))(
        segment["next_observation"], initial_state, segment["start"], segment["next_terminated"], key
    ))
    # Double DQN
    next_q = next_q.squeeze(1)[batch_index, time_index, next_q_action_idx.squeeze(1).argmax(-1).flatten()].reshape(B, T)

    target = segment["next_reward"] + (1.0 - segment["next_terminated"]) * gamma * next_q
    error = selected_q - target
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(masked_mean(jnp.abs(error), segment['mask']))
    q_mean = masked_mean(q_values.squeeze(1), jnp.expand_dims(segment['mask'], -1))
    target_mean = masked_mean(target, segment['mask'])
    target_network_mean = masked_mean(next_q, segment['mask'])
    return loss, (q_mean, target_mean, target_network_mean, error_min, error_max)


def segment_dqn_loss(q_network, q_target, segment, gamma, key):
    B, T = segment["next_reward"].shape
    initial_state = q_network.initial_state()
    q_values, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0, None))(
        segment["observation"], initial_state, segment["start"], segment["next_terminated"], key
    )
    batch_index = jnp.repeat(jnp.arange(B), T)
    time_index = jnp.tile(jnp.arange(T), B)
    # B, ensemble, T, action
    selected_q = q_values.squeeze(1)[
        batch_index, time_index, segment["action"].reshape(-1)
    ].reshape(segment["action"].shape)

    next_q, _ = jax.lax.stop_gradient(eqx.filter_vmap(q_target, in_axes=(0, None, 0, 0, None))(
        segment["next_observation"], initial_state, segment["start"], segment["next_terminated"], key
    ))
    next_q = next_q.squeeze(1)[batch_index, time_index].argmax(-1).reshape(B, T)

    target = segment["next_reward"] + (1.0 - segment["next_terminated"]) * gamma * next_q
    error = selected_q - target
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(masked_mean(jnp.abs(error), segment['mask']))
    q_mean = masked_mean(q_values.squeeze(1), jnp.expand_dims(segment['mask'], -1))
    target_mean = masked_mean(target, segment['mask'])
    target_network_mean = masked_mean(next_q, segment['mask'])
    return loss, (q_mean, target_mean, target_network_mean, error_min, error_max)


def tape_ddqn_loss_filtered(obs, q_network, q_target, tape, gamma, key):
    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)
    initial_state = q_network.initial_state()
    q_values, _ = q_network(
        obs, initial_state, tape["start"], tape["next_done"], key=key
    )
    batch_index = jnp.arange(B)
    selected_q = q_values.squeeze(0)[batch_index, tape["action"]]
    error = selected_q[-1]
    return jnp.abs(error)


def tape_dqn_loss(q_network, q_target, tape, gamma, key):
    B = tape["next_reward"].shape[0]
    initial_state = q_network.initial_state()
    q_values, _ = q_network(
        tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
    )
    batch_index = jnp.arange(B)
    # Extra singleton ensemble dim in front 
    selected_q = q_values.squeeze(0)[batch_index, tape["action"]]

    next_q_target, _ = jax.lax.stop_gradient(q_target(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    # Extra singleton ensemble dim in front 
    next_q = next_q_target.squeeze(0).max(-1)

    done = jnp.logical_or(tape['next_terminated'], tape['next_truncated'])
    target = tape["next_reward"] + (1.0 - done) * gamma * next_q 
    error = selected_q - target
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(error)
    loss = loss.mean()
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss, (q_mean, target_mean, target_network_mean, error_min, error_max)


@partial(eqx.filter_value_and_grad, has_aux=True)
def online_tape_redq_loss(q_network, q_target, tape, gamma, ensemble_subset_size, progress, key):
    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)
    initial_state = q_network.initial_state()
    q_values, _ = q_network(
        tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
    )
    selected_q = q_values[:, batch_idx, tape["action"]]

    next_q_target, _ = jax.lax.stop_gradient(q_target(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    _, k = jax.random.split(key, 2)
    next_q = jnp.median(jax.random.choice(k, next_q_target, (ensemble_subset_size,), replace=False).max(-1), axis=0)

    done = jnp.logical_or(tape['next_terminated'], tape['next_truncated'])
    target = tape["next_reward"] + (1.0 - done) * gamma * next_q 
    error = selected_q - target
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(error)
    loss = loss.mean()
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss, (q_mean, target_mean, target_network_mean, error_min, error_max)

def tape_ddqn_loss(q_network, q_target, tape, gamma, key):
    # Only changes from standard DDQN are computing the Q values using a recurrent Q network
    # We have an additional dimension for ensembles (not used in the paper): [ensemble_member, num_transitions]
    # This is why there are .squeeze(0) everywhere
    initial_state = q_network.initial_state() # This should be the only difference compared to nonrecurrent Q function
    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)
    batch_index = jnp.arange(B)

    q_values, _ = q_network(
        tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
    )
    selected_q = q_values.squeeze(0)[batch_index, tape["action"]]

    next_q_action_idx, _ = jax.lax.stop_gradient(q_network(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    next_q, _ = jax.lax.stop_gradient(q_target(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    next_q = next_q.squeeze(0)[batch_idx, next_q_action_idx.argmax(-1).flatten()] 

    target = tape["next_reward"] + (1.0 - tape["next_terminated"]) * gamma * next_q 
    error = selected_q - target
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(error)
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean, error_min, error_max)
    


def tape_update(q_network, q_target, data, opt, opt_state, gamma, tau, loss_key):
    outputs, gradient = eqx.filter_value_and_grad(tape_ddqn_loss, has_aux=True)(
        q_network, q_target, data, gamma, loss_key
    )
    loss, (q_mean, target_mean, target_network_mean, error_min, error_max) = outputs
    updates, opt_state = opt.update(
        gradient, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)
    return q_network, q_target, opt_state, q_mean, target_mean, target_network_mean, error_min, error_max, loss, gradient


def segment_update(q_network, q_target, data, opt, opt_state, gamma, tau, loss_key):
    outputs, gradient = eqx.filter_value_and_grad(segment_ddqn_loss, has_aux=True)(
        q_network, q_target, data, gamma, loss_key
    )
    loss, (q_mean, target_mean, target_network_mean, error_min, error_max) = outputs
    updates, opt_state = opt.update(
        gradient, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)
    return q_network, q_target, opt_state, q_mean, target_mean, target_network_mean, error_min, error_max, loss, gradient
