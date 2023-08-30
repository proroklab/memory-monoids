from functools import partial
import jax
from jax import numpy as jnp
import equinox as eqx


@partial(eqx.filter_value_and_grad, has_aux=True)
def segment_constrained_dqn_loss(q_network, q_target, segment, gamma):
    B, T = segment["reward"].shape
    initial_state = q_network.initial_state()
    q_values, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0))(
        segment["observation"], initial_state, segment["start"], segment["done"]
    )
    batch_index = jnp.repeat(jnp.arange(B), T)
    time_index = jnp.tile(jnp.arange(T), B)
    selected_q = q_values[
        batch_index, time_index, segment["action"].reshape(-1)
    ].reshape(segment["action"].shape)
    next_q_target, _ = eqx.filter_vmap(q_target, in_axes=(0, None, 0, 0))(
        segment["next_observation"], initial_state, segment["start"], segment["done"]
    )
    next_q, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0))(
        segment["next_observation"], initial_state, segment["start"], segment["done"]
    )
    next_q = jnp.minimum(next_q, next_q_target)
    target = jax.lax.stop_gradient(
        segment["reward"] + (1.0 - segment["done"]) * gamma * next_q.max(-1)
    )
    error = selected_q - target
    # Cannot jit the loss due to masking
    loss = jnp.abs(error[segment["mask"]])
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean)


@partial(eqx.filter_value_and_grad, has_aux=True)
def segment_dqn_loss(q_network, q_target, segment, gamma):
    B, T = segment["reward"].shape
    initial_state = q_network.initial_state()
    q_values, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0))(
        segment["observation"], initial_state, segment["start"], segment["done"]
    )
    batch_index = jnp.repeat(jnp.arange(B), T)
    time_index = jnp.tile(jnp.arange(T), B)
    selected_q = q_values[
        batch_index, time_index, segment["action"].reshape(-1)
    ].reshape(segment["action"].shape)
    next_q, _ = eqx.filter_vmap(q_target, in_axes=(0, None, 0, 0))(
        segment["next_observation"], initial_state, segment["start"], segment["done"]
    )
    target = jax.lax.stop_gradient(
        segment["reward"] + (1.0 - segment["done"]) * gamma * next_q.max(-1)
    )
    error = selected_q - target
    # Cannot jit the loss due to masking
    loss = jnp.abs(error[segment["mask"]])
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean)


@eqx.filter_jit
@partial(eqx.filter_value_and_grad, has_aux=True)
def stream_dqn_loss(q_network, q_target, stream, gamma):
    B = stream["reward"].shape[0]
    initial_state = q_network.initial_state()
    q_values, _ = q_network(
        stream["observation"], stream["start"], stream["done"], initial_state
    )
    batch_index = jnp.arange(B)
    selected_q = q_values[batch_index, stream["action"]]
    next_q, _ = q_target(
        stream["next_observation"], stream["start"], stream["done"], initial_state
    )
    target = jax.lax.stop_gradient(
        stream["reward"] + (1.0 - stream["done"]) * gamma * next_q.max(-1)
    )
    error = selected_q - target
    loss = jnp.abs(error)
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean)


@partial(eqx.filter_value_and_grad, has_aux=True)
def dqn_loss_1d(q_network, q_target, segment):
    T = segment["reward"].shape[0]
    initial_state = q_network.initial_state()
    q_values, _ = q_network(segment["observation"], initial_state)
    time_index = jnp.arange(T)
    selected_q = q_values[time_index, segment["action"]]
    next_q, _ = q_target(segment["next_observation"], initial_state)
    target = segment["reward"] + (1.0 - segment["done"]) * config["train"][
        "gamma"
    ] * next_q.max(-1)
    error = selected_q - jax.lax.stop_gradient(target)
    # Cannot jit the loss due to masking
    # loss = jnp.abs(error[segment["mask"]])
    loss = error[segment["mask"]] ** 2
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean)
