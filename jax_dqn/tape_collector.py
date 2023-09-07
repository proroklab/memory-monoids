from typing import Dict, List, NamedTuple
import numpy as np
from jax import random


# Done is true at the initial state of the following episode
# E.g.
# env.step() -> (observation, False)
# policy(observation, False)
# # Terminal state
# env.step() -> (observation, True)
# env.reset()
# Env is reset before the policy ever sees the "done" state

# However, the way we store, we will append 'done' to the end of the episode
# so 'done' will align with the last action the agent took
# it really means "done after this action"
# marking the last transition in the episode
# q(next_obs) * done
#
# Start will mark the first transition of the episode

## ISSUE: collector done is offset from training done


class TapeCollector:
    def __init__(self, env, config):
        self.env = env
        self.config = config["collect"]
        self.obs_shape = env.observation_space.shape[0]
        self.act_shape = env.action_space.n

        self.observation = None
        self.next_observation = None
        self.action = None
        self.reward = None
        self.done = True
        self.start = True
        self.sampled_frames = 0
        self.sampled_epochs = 0
        self.episode_id = 0
        self.episode_rewards = []
        self.running_reward = 0
        self.best_reward = -np.inf

    def __call__(self, q_network, policy, progress, key, need_reset=False):
        observations = np.zeros(
            (self.config["steps_per_epoch"], self.obs_shape), np.float32
        )
        actions = np.zeros((self.config["steps_per_epoch"]), np.int32)
        rewards = np.zeros((self.config["steps_per_epoch"]), np.float32)
        next_observations = np.zeros(
            (self.config["steps_per_epoch"], self.obs_shape), dtype=np.float32
        )
        dones = np.zeros((self.config["steps_per_epoch"]), dtype=bool)
        starts = np.zeros((self.config["steps_per_epoch"]), dtype=bool)

        if need_reset:
            self.done = self.next_done = False
            key, reset_key = random.split(key)
            self.observation, _ = self.env.reset(
                seed=random.bits(reset_key).item()
            )
            self.recurrent_state = q_network.initial_state()
            self.reward = 0
            self.running_reward = 0
            self.episode_id += 1

        for step in range(self.config["steps_per_epoch"]):
            if self.done:
                self.done = self.next_done = False
                key, reset_key = random.split(key)
                self.observation, _ = self.env.reset(
                    seed=random.bits(reset_key).item()
                )
                self.recurrent_state = q_network.initial_state()
                self.reward = 0
                self.episode_rewards.append(self.running_reward)
                self.running_reward = 0


            if self.sampled_epochs < self.config["random_epochs"]:
                self.action = self.env.action_space.sample()
            else:
                key, action_key = random.split(key)
                self.action, self.recurrent_state = policy(
                    q_network=q_network,
                    x=self.observation,
                    state=self.recurrent_state,
                    start=np.array([self.start]),
                    done=np.array([self.done]),
                    progress=progress,
                    epsilon_start=self.config["eps_start"],
                    epsilon_end=self.config["eps_end"],
                    key=action_key,
                )
                self.action = self.action.item()
            (
                self.next_observation,
                self.reward,
                terminated,
                truncated,
                _,
            ) = self.env.step(self.action)
            self.done = terminated or truncated
            self.running_reward += self.reward

            observations[step] = self.observation
            actions[step] = self.action
            rewards[step] = self.reward
            next_observations[step] = self.next_observation
            starts[step] = self.start
            dones[step] = self.done
            self.observation = self.next_observation
            self.start = False

        transitions = {
            "observation": observations, 
            "action": actions, 
            "reward": rewards, 
            "next_observation": next_observations, 
            "start": starts, 
            "done": dones, 
        }
        # Last episode will be truncated
        self.sampled_frames += step
        self.sampled_epochs += 1
        mean_reward = np.mean(self.episode_rewards)
        self.episode_rewards = []
        if mean_reward == np.inf:
            mean_reward = -np.inf
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
        return (
            transitions,
            mean_reward,
            self.best_reward
        )
