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


class SegmentCollector:
    # Lifetime variables
    sampled_frames = 0
    sampled_epochs = 0
    episode_id = 0
    best_reward = -np.inf

    # Discarded state between calls
    episode_reward = []
    seq_lens = []
    seq_len = 0

    def __init__(self, env, config):
        self.env = env
        self.config = config["collect"]
        self.obs_shape = env.observation_space.shape[0]
        self.act_shape = env.action_space.n

    def simple_split_and_pad(self, arrays: Dict[str, np.ndarray], seq_len):
        # First split at episode boundaries
        max_len = self.config["segment_length"]
        arrays['mask'] = np.ones_like(arrays['start'])
        output = {}
        for name, array in arrays.items():
            #episodes = np.array_split(array, episode_offsets)
            episode = array[:seq_len]
            fragments = [
                frag 
                for frag in np.array_split(episode, np.arange(max_len, episode.shape[0], max_len)) 
            ]
            # Pad fragments into segments
            segments = [
                np.concatenate([frag, np.zeros((max_len - len(frag), *frag.shape[1:]), dtype=frag.dtype)], axis=0)
                for frag in fragments
            ]
            output[name] = np.stack(segments, axis=0)
        return output

    def split_and_pad(self, arrays: Dict[str, np.ndarray], seq_lens):
        # First split at episode boundaries
        max_len = self.config["segment_length"]
        # Add on the final sequence idx (this should be equal to segment_length)
        #seq_lens = seq_lens + [max_len - seq_lens[-1]]
        episode_offsets = np.cumsum(seq_lens)
        arrays['mask'] = np.ones_like(arrays['done'])
        output = {}
        for name, array in arrays.items():
            episodes = np.array_split(array, episode_offsets)
            fragments = [
                frag 
                for episode in episodes
                for frag in np.array_split(episode, np.arange(max_len, episode.shape[0], max_len)) 
                if len(frag) > 0
            ]
            # Pad fragments into segments
            segments = [
                np.concatenate([frag, np.zeros((max_len - len(frag), *frag.shape[1:]), dtype=frag.dtype)], axis=0)
                for frag in fragments
            ]
            output[name] = np.stack(segments, axis=0)
        return output

    def __call__(self, q_network, policy, progress, key, need_reset=False):
        observations = []
        actions = []
        next_rewards = []
        terminateds = []
        truncateds = []
        starts = []
        episode_ids = []

        key, reset_key = random.split(key)
        observation, _ = self.env.reset(seed=random.bits(reset_key).item())
        terminated = False
        truncated = False
        start = True

        observations.append(observation)
        terminateds.append(terminated)
        truncateds.append(truncated)
        starts.append(start)

        running_reward = 0
        self.episode_id += 1
        recurrent_state = q_network.initial_state()
        step = 0
        while not (terminated or truncated):
            if self.sampled_epochs < self.config["random_epochs"]:
                action = self.env.action_space.sample()
            else:
                key, action_key = random.split(key)
                action, recurrent_state = policy(
                    q_network=q_network,
                    x=observation,
                    state=recurrent_state,
                    start=np.array([start]),
                    done=np.array([terminated or truncated]),
                    progress=progress,
                    epsilon_start=self.config["eps_start"],
                    epsilon_end=self.config["eps_end"],
                    key=action_key,
                )
                action = action.item()
            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = self.env.step(action)
            start = False

            observations.append(observation)
            terminateds.append(terminated)
            truncateds.append(truncated)
            starts.append(False)
            actions.append(action)
            episode_ids.append(self.episode_id)
            next_rewards.append(reward)

            running_reward += reward
            step += 1

        transitions = {
            "observation": np.array(observations[:-1]), 
            "action": np.array(actions), 
            "next_reward": np.array(next_rewards), 
            "next_observation": np.array(observations[1:]),
            "next_done": np.array(terminateds[1:]) + np.array(truncateds[1:]),
            "next_terminated": np.array(terminateds[1:]),
            "next_truncated": np.array(truncateds[1:]),
            "start": np.array(starts[:-1]), 
            "mask": np.ones(len(next_rewards), dtype=bool),
            "episode_id": np.array(episode_ids),
        }
        # Validate
        # for k in ['observation', 'done']:
        #     masked = transitions[k][transitions['mask']]
        #     masked_next = transitions['next_' + k][transitions['mask']]
        #     if not np.all(masked[1:] == masked_next[:-1]):
        #         raise RuntimeError("Collector did something naughty")

        # for k in transitions:
        #     transitions[k] = np.concatenate([
        #         transitions[k], 
        #         np.zeros(
        #             shape=(self.config['steps_per_epoch'] - len(transitions[k]), *transitions[k].shape[1:]), 
        #             dtype=transitions[k].dtype)
        #     ], axis=0)

        transitions = self.simple_split_and_pad(transitions, len(actions))


        self.sampled_frames += step
        self.sampled_epochs += 1
        if running_reward > self.best_reward:
            self.best_reward = running_reward
        return (
            transitions,
            running_reward, 
            self.best_reward
        )