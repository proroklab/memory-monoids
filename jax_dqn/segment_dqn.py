from functools import partial
from jax import numpy as jnp
from jax import vmap, jit, random
import jax
import numpy as np
from jax import random, vmap, nn
from cpprb import ReplayBuffer

# import flax
# from flax import linen as nn
import equinox as eqx
from equinox import nn
import gymnasium
from modules import greedy_policy
from modules import hard_update
import popgym
import optax
import tqdm
import wandb

from modules import QNetwork, epsilon_greedy_policy, anneal


#env = popgym.envs.StatelessCartPole()
env = gymnasium.make('CartPole-v0')
eval_env = gymnasium.make('CartPole-v0')
#eval_env = popgym.envs.StatelessCartPole()

segment_length = 80
eval_length = 201
obs_shape = env.observation_space.shape[0]
act_shape = env.action_space.n
num_epochs = 3_000
target_delay = 100
prefill_epochs = 500
buffer_num_segments = num_epochs + prefill_epochs
batch_size = 128
gamma = 0.99
seed = 0
eval_seed = 1000
eps_start = 0.3
eps_end = 0.01
mlp_size = 64
recurrent_size = 64
eval_interval = 10
key = random.PRNGKey(seed)
eval_key = random.PRNGKey(eval_seed)
# opt = optax.chain(
#     optax.clip(1.0),
#     optax.adamw(1e-3)
# )
opt = optax.adamw(1e-3)


@eqx.filter_value_and_grad
def dqn_loss(q_network, q_target, segment):
    B, T = segment["reward"].shape
    initial_state = q_network.initial_state()
    q_values, _ = eqx.filter_vmap(q_network, in_axes=(0, None))(segment["observation"], initial_state)
    batch_index = jnp.repeat(jnp.arange(B), T)
    time_index = jnp.tile(jnp.arange(T), B)
    selected_q = q_values[
       batch_index, time_index, segment["action"].reshape(-1)
    ].reshape(segment["action"].shape)
    next_q, _ = eqx.filter_vmap(q_target, in_axes=(0, None))(segment["next_observation"], initial_state)
    target = jax.lax.stop_gradient(
        segment["reward"] + (1.0 - segment["done"]) * gamma * next_q.max(-1)
    )
    error = selected_q - target
    # Cannot jit the loss due to masking
    loss = jnp.abs(error[segment["mask"]])
    return loss.mean()

import time
start = time.time()

@partial(eqx.filter_value_and_grad, has_aux=True)
def dqn_loss_1d(q_network, q_target, segment):
    T = segment["reward"].shape[0]
    initial_state = q_network.initial_state()
    q_values, _ = q_network(segment["observation"], initial_state)
    time_index = jnp.arange(T)
    selected_q = q_values[
       time_index, segment["action"]
    ]
    next_q, _ = q_target(segment["next_observation"], initial_state)
    target = segment["reward"] + (1.0 - segment["done"]) * gamma * next_q.max(-1)
    error = selected_q - jax.lax.stop_gradient(target)
    # Cannot jit the loss due to masking
    #loss = jnp.abs(error[segment["mask"]])
    loss = error[segment["mask"]] ** 2
    q_mean = jnp.mean(q_values)
    target_mean = jnp.mean(target)
    target_network_mean = jnp.mean(next_q)
    return loss.mean(), (q_mean, target_mean, target_network_mean)


rb = ReplayBuffer(
    buffer_num_segments,
    {
        "observation": {"shape": (segment_length, obs_shape), "dtype": jnp.float32},
        "action": {"shape": (segment_length), "dtype": jnp.int32},
        "reward": {"shape": (segment_length), "dtype": jnp.float32},
        "next_observation": {
            "shape": (segment_length, obs_shape),
            "dtype": jnp.float32,
        },
        "done": {"shape": (segment_length), "dtype": bool},
        "mask": {"shape": (segment_length), "dtype": bool},
    },
)


class SegmentCollector:
    def __init__(self, env, segment_length, random_epochs=0, propagate_state=False):
        self.env = env
        self.segment_length = segment_length
        self.propagate_state = propagate_state
        self.random_epochs = random_epochs

        self.observation = None
        self.next_observation = None
        self.action = None
        self.reward = None
        self.terminated = self.truncated = True
        self.sampled_frames = 0
        self.sampled_epochs = 0
        self.episode_reward = 0

    def __call__(self, q_network, policy, progress, key, need_reset=False):
        observations = np.zeros((self.segment_length, obs_shape), np.float32)
        actions = np.zeros((self.segment_length), np.int32)
        rewards = np.zeros((self.segment_length), np.float32)
        next_observations = np.zeros((self.segment_length, obs_shape), dtype=np.float32)
        dones = np.zeros((self.segment_length), dtype=bool)
        mask = np.ones((self.segment_length), dtype=bool)

        if self.terminated or self.truncated or need_reset:
            key, reset_key = random.split(key)
            self.observation, _ = env.reset(seed=random.bits(reset_key).item())
            self.recurrent_state = q_network.initial_state()
            self.episode_reward = 0

        self.terminated = self.truncated = False

        episode_reward = -np.inf
        action_keys = random.split(key, self.segment_length)
        for step in range(self.segment_length):
            if self.sampled_epochs < self.random_epochs:
                self.action = env.action_space.sample()
            else:
                self.action, self.recurrent_state = policy(
                    q_network=q_network,
                    x=self.observation,
                    state=self.recurrent_state,
                    progress=progress,
                    epsilon_start=eps_start,
                    epsilon_end=eps_end,
                    key=action_keys[step],
                )
                self.action = self.action.item()
            self.next_observation, self.reward, self.terminated, self.truncated, _ = env.step(self.action)
            observations[step] = self.observation
            actions[step] = self.action
            rewards[step] = self.reward
            next_observations[step] = self.next_observation
            dones[step] = self.terminated or self.truncated
            self.observation = self.next_observation

            self.episode_reward += self.reward
            done = self.terminated or self.truncated
            if done:
                mask[step + 1:] = False
                episode_reward = self.episode_reward
                self.episode_reward = 0
                break
        
        if not self.propagate_state:
            self.recurrent_state = q_network.initial_state()

        self.sampled_frames += step + 1
        self.sampled_epochs += 1
        self.episode_reward += jnp.sum(rewards)
        return observations, actions, rewards, next_observations, dones, mask, episode_reward


key, *keys = random.split(key, 3)
q_network = QNetwork(obs_shape, mlp_size, recurrent_size, act_shape, keys[0])
q_target = QNetwork(obs_shape, mlp_size, recurrent_size, act_shape, keys[0])
opt_state = opt.init(eqx.filter(q_network, eqx.is_inexact_array))
epochs = tqdm.tqdm(range(1, prefill_epochs + num_epochs + 1))
best_eval_reward = best_cumulative_reward = eval_reward = -np.inf
need_reset = True
collector = SegmentCollector(env, segment_length, prefill_epochs)
eval_collector = SegmentCollector(eval_env, eval_length)
for epoch in epochs:
    key, epoch_key = random.split(key)
    progress = max(0, (epoch - prefill_epochs) / num_epochs)
    (
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        mask,
        cumulative_reward,
    ) = collector(q_network, epsilon_greedy_policy, progress, epoch_key, False)
    if cumulative_reward > best_cumulative_reward:
        best_cumulative_reward = cumulative_reward

    rb.add(
        observation=observations,
        action=actions,
        reward=rewards,
        next_observation=next_observations,
        done=dones,
        mask=mask,
    )
    rb.on_episode_end()

    if epoch <= prefill_epochs:
        continue

    data = rb.sample(1)
    #loss, gradient = dqn_loss(q_network, q_target, data)
    data = jax.tree_util.tree_map(lambda x: x.squeeze(0), data)
    #(loss, gradient), (q_mean, target_mean, target_network_mean) = 
    outputs, gradient = dqn_loss_1d(q_network, q_target, data)
    loss, (q_mean, target_mean, target_network_mean) = outputs
    updates, opt_state = opt.update(
        gradient, opt_state, params=eqx.partition(q_network, eqx.is_inexact_array)[0]
    )
    q_network = eqx.apply_updates(q_network, updates)

    if epoch % target_delay == 0:
        q_target = hard_update(q_network, q_target)

    # Eval
    if epoch % eval_interval == 0:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            eval_reward,
        ) = collector(q_network, greedy_policy, 1.0, epoch_key, True)
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward

    to_log = {
        "eval/reward": eval_reward,
        "train/reward": cumulative_reward,
        "train/loss": loss,
        "train/epsilon": anneal(eps_start, eps_end, progress),
        "train/buffer_capacity": rb.get_stored_size() / buffer_num_segments,
        "train/q_mean": q_mean,
        "train/target_mean": target_mean,
        "train/target_network_mean": target_network_mean,
    }

    epochs.set_description(
        f"eval{eval_reward:.2f}, {best_eval_reward:.2f} "
        + f"train: {cumulative_reward:.2f}, {best_cumulative_reward:.2f} "
        + f"loss: {loss:.3f} "
        + f"eps: {anneal(eps_start, eps_end, progress):.2f} "
        + f"buf: {rb.get_stored_size() / buffer_num_segments:.2f} "
        + f"qm: {q_mean:.2f} "
        + f"tm: {target_mean:.2f} "
        + f"qtm: {target_network_mean:.2f} "
    )
