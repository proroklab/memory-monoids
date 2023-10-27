from functools import partial
import jax
from jax import numpy as jnp
import equinox as eqx

from modules import softsymlog


def huber(x):
    return jax.lax.select(
        jnp.abs(x) < 1.0,
        0.5 * x ** 2,
        jnp.abs(x) - 0.5 
    )

def masked_mean(x, mask):
    return jnp.sum(x * mask) / mask.sum()

@partial(eqx.filter_value_and_grad, has_aux=True)
def segment_ddqn_loss(q_network, q_target, segment, gamma, key):
    # Double DQN
    B, T = segment["next_reward"].shape
    initial_state = q_network.initial_state()
    q_values, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0, None))(
        segment["observation"], initial_state, segment["start"], segment["next_terminated"], key
    )
    batch_index = jnp.repeat(jnp.arange(B), T)
    time_index = jnp.tile(jnp.arange(T), B)
    selected_q = q_values[
        batch_index, time_index, segment["action"].reshape(-1)
    ].reshape(segment["action"].shape)

    next_q_action_idx, _ = eqx.filter_vmap(q_network, in_axes=(0, None, 0, 0, None))(
        segment["next_observation"], initial_state, segment["start"], segment["next_terminated"], key
    )
    next_q, _ = jax.lax.stop_gradient(eqx.filter_vmap(q_target, in_axes=(0, None, 0, 0, None))(
        segment["next_observation"], initial_state, segment["start"], segment["next_terminated"], key
    ))
    # Double DQN
    next_q = next_q[batch_index, time_index, next_q_action_idx.argmax(-1).flatten()].reshape(B, T)

    target = segment["next_reward"] + (1.0 - segment["next_terminated"]) * gamma * next_q
    error = selected_q - target
    error = jnp.clip(error, a_max=2 * error.std())
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(masked_mean(jnp.abs(error), segment['mask']))
    q_mean = masked_mean(q_values, jnp.expand_dims(segment['mask'], -1))
    target_mean = masked_mean(target, segment['mask'])
    target_network_mean = masked_mean(next_q, segment['mask'])
    return loss, (q_mean, target_mean, target_network_mean, error_min, error_max)


def tempered_softmax(x, temp=10, key=None):
    return jax.nn.softmax(x * temp)


def tape_constrained_q_loss(q_network, tape, gamma, key):
    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)
    initial_state = q_network.initial_state()

    def l1(q_network):
        q_values, _ = q_network(
            tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
        )
        batch_index = jnp.arange(B)
        selected_q = q_values[batch_index, tape["action"]]

        next_q, _ = jax.lax.stop_gradient(q_network(
            tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
        ))
        next_q = next_q.max(-1)
        target = tape["next_reward"] + (1.0 - tape["next_terminated"]) * gamma * next_q 
        error = selected_q - target
        loss = huber(error).mean()
        return loss

    def l2(q_network):
        q_values, _ = jax.lax.stop_gradient(q_network(
            tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
        ))
        batch_index = jnp.arange(B)
        selected_q = q_values[batch_index, tape["action"]]

        next_q, _ = q_network(
            tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
        )
        next_q = next_q.max(-1)
        target = tape["next_reward"] + (1.0 - tape["next_terminated"]) * gamma * next_q 
        error = selected_q - target
        loss = huber(error).mean()
        return loss

    g1 = eqx.filter_grad(l1)(q_network)
    g2 = eqx.filter_grad(l2)(q_network)
    g_hat = g2 / (1e-8 + jnp.linalg.norm(g2, keepdims=True))
    g_pi = jnp.cross(g * g_hat, g_hat)
    g_cons = g1 - g_pi
    return g_cons, (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))



@partial(eqx.filter_value_and_grad, has_aux=True)
def online_tape_q_loss(q_network, q_target, tape, gamma, key):
    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)
    initial_state = q_network.initial_state()
    q_values, _ = q_network(
        tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
    )
    batch_index = jnp.arange(B)
    selected_q = q_values[batch_index, tape["action"]]

    _, k0, k1 = jax.random.split(key, 3)
#    noise0 = jax.random.normal(key, shape=tape["next_observation"].shape) * 0.001
#    noise1 = jax.random.normal(key, shape=tape["next_observation"].shape) * 0.001
    next_q_action_idx, _ = q_network(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    )
    next_q_target, _ = jax.lax.stop_gradient(q_target(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    #next_q = jax.lax.stop_gradient(jnp.minimum(next_q_target, next_q_action_idx)[batch_idx, next_q_action_idx.argmax(-1).flatten()] )
#    diff = huber(next_q_action_idx - next_q_target)
#    diff = diff / (1e-6 + jnp.max(diff))
#    next_q = jax.lax.stop_gradient((diff * next_q_target + (1.0 - diff) * next_q_action_idx)[batch_idx, next_q_action_idx.argmax(-1).flatten()])
    next_q = jax.lax.stop_gradient(next_q_target[batch_idx, next_q_action_idx.argmax(-1).flatten()])

    target = tape["next_reward"] + (1.0 - tape["next_terminated"]) * gamma * next_q 
    error = selected_q - target
    # Objective should be softmax(difference) to prevent a single q from exploding
    # Clip large positive errors
    #error = jnp.clip(error, a_max=2 * error.std())
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(error)
    #constraint = 0.05 * jax.nn.logsumexp(loss - jnp.log(B))
    # Mean is across action dim
    #constraint = jax.nn.logsumexp(huber(next_q_action_idx - next_q_target) - jnp.log(B))
    #constraint = jax.scipy.special.kl_div(jax.nn.softmax(next_q_action_idx, axis=-1), jax.nn.softmax(next_q_target, axis=-1)).sum(-1)
    # TODO: Constrain latent state variance instead of q values
    #constraint = huber(next_q_action_idx - next_q_target)
    #constraint = 0.0001 * huber(jax.nn.logsumexp(next_q_action_idx - next_q_target) - jnp.log(next_q_target.size))
    #constraint = huber(jax.nn.softmax(next_q_action_idx, axis=-1) - jax.nn.softmax(next_q_target, axis=-1))
    loss = loss.mean() #+ constraint.mean() 
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss, (q_mean, target_mean, target_network_mean, error_min, error_max)

@partial(eqx.filter_value_and_grad, has_aux=True)
def tape_ddqn_loss(q_network, q_target, tape, gamma, key):
    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)
    initial_state = q_network.initial_state()
    q_values, _ = q_network(
        tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
    )
    batch_index = jnp.arange(B)
    selected_q = q_values[batch_index, tape["action"]]

    next_q_action_idx, _ = jax.lax.stop_gradient(q_network(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    next_q, _ = jax.lax.stop_gradient(q_target(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    next_q = next_q[batch_idx, next_q_action_idx.argmax(-1).flatten()] 

    target = tape["next_reward"] + (1.0 - tape["next_terminated"]) * gamma * next_q 
    error = selected_q - target
    # Clip large positive errors
    error = jnp.clip(error, a_max=2 * error.std())
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(error)
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean, error_min, error_max)

@partial(eqx.filter_value_and_grad, has_aux=True)
def tape_ddqn_loss(q_network, q_target, tape, gamma, key):
    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)
    initial_state = q_network.initial_state()
    q_values, _ = q_network(
        tape["observation"], initial_state, tape["start"], tape["next_done"], key=key
    )
    batch_index = jnp.arange(B)
    selected_q = q_values[batch_index, tape["action"]]

    next_q_action_idx, _ = jax.lax.stop_gradient(q_network(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    next_q, _ = jax.lax.stop_gradient(q_target(
        tape["next_observation"], initial_state, tape["start"], tape["next_done"], key
    ))
    next_q = next_q[batch_idx, next_q_action_idx.argmax(-1).flatten()] 

    target = tape["next_reward"] + (1.0 - tape["next_terminated"]) * gamma * next_q 
    error = selected_q - target
    # Clip large positive errors
    error = jnp.clip(error, a_max=2 * error.std())
    error_min, error_max = jnp.min(error), jnp.max(error)
    loss = huber(error)
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean, error_min, error_max)
