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


class SegmentCollector:
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
        self.episode_reward = 0

    def __call__(self, q_network, policy, progress, key, need_reset=False):
        observations = np.zeros(
            (self.config["segment_length"], self.obs_shape), np.float32
        )
        actions = np.zeros((self.config["segment_length"]), np.int32)
        rewards = np.zeros((self.config["segment_length"]), np.float32)
        next_observations = np.zeros(
            (self.config["segment_length"], self.obs_shape), dtype=np.float32
        )
        dones = np.zeros((self.config["segment_length"]), dtype=bool)
        starts = np.zeros((self.config["segment_length"]), dtype=bool)
        mask = np.ones((self.config["segment_length"]), dtype=bool)

        if self.done or need_reset:
            key, reset_key = random.split(key)
            self.done = False
            self.observation, self.start, _ = self.env.reset(
                seed=random.bits(reset_key).item()
            )
            self.recurrent_state = q_network.initial_state()
            self.episode_reward = 0

        episode_reward = -np.inf
        action_keys = random.split(key, self.config["segment_length"])
        for step in range(self.config["segment_length"]):
            if self.sampled_epochs < self.config["random_epochs"]:
                self.action = self.env.action_space.sample()
            else:
                self.action, self.recurrent_state = policy(
                    q_network=q_network,
                    x=self.observation,
                    state=self.recurrent_state,
                    start=self.start,
                    done=self.done,
                    progress=progress,
                    epsilon_start=self.config["eps_start"],
                    epsilon_end=self.config["eps_end"],
                    key=action_keys[step],
                )
                self.action = self.action.item()
            (
                self.next_observation,
                self.reward,
                terminated,
                truncated,
                self.next_start,
                _,
            ) = self.env.step(self.action)
            self.done = terminated or truncated
            observations[step] = self.observation
            dones[step] = self.done
            actions[step] = self.action
            rewards[step] = self.reward
            next_observations[step] = self.next_observation
            dones[step] = self.done
            starts[step] = self.start
            self.observation = self.next_observation
            self.start = self.next_start

            self.episode_reward += self.reward
            if self.done:
                mask[step + 1 :] = False
                episode_reward = self.episode_reward
                self.episode_reward = 0
                break

        if not self.config["propagate_state"]:
            self.recurrent_state = q_network.initial_state()

        self.sampled_frames += step + 1
        self.sampled_epochs += 1
        return (
            observations,
            actions,
            rewards,
            next_observations,
            starts,
            dones,
            mask,
            episode_reward,
        )
