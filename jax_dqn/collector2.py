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


class BatchedSegmentCollector:
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
        self.episode_reward = {}
        self.best_reward = -np.inf

    def update_episodic_reward(self, padded_reward, padded_episode_id):
        segment_rewards = np.cumsum(padded_reward, axis=1)[:,-1]
        episode_ids = padded_episode_id[:, 0]

        for e_id, reward in zip(episode_ids, segment_rewards): 
            self.episode_reward.update(
                {e_id: reward}
            )

    def get_episodic_reward(self, padded_episode_id, done):
        done_segments = np.any(done, axis=1)
        id_segments = padded_episode_id[:,0]
        segments_to_mean = id_segments[done_segments]
        reward = []
        for s_id in segments_to_mean:
            reward.append(self.episode_reward.pop(s_id))
        if len(reward) == 0:
            return -np.inf
        return np.mean(reward)

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
        episode_id = -1 * np.ones((self.config["steps_per_epoch"]), np.int64)
        mask = np.zeros((self.config["steps_per_epoch"]), dtype=bool)

        seq_len = 0
        seq_lens = []

        if need_reset:
            self.done = self.next_done = False
            key, reset_key = random.split(key)
            self.observation, self.start, _ = self.env.reset(
                seed=random.bits(reset_key).item()
            )
            self.recurrent_state = q_network.initial_state()
            seq_lens.append(seq_len)
            seq_len = 0
            self.reward = 0
            self.episode_id += 1

        for step in range(self.config["steps_per_epoch"]):
            if self.done:
                self.done = self.next_done = False
                key, reset_key = random.split(key)
                self.observation, self.start, _ = self.env.reset(
                    seed=random.bits(reset_key).item()
                )
                self.recurrent_state = q_network.initial_state()
                seq_lens.append(seq_len)
                seq_len = 0
                self.reward = 0
                self.episode_id += 1

            seq_len += 1

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
                _, #self.next_start,
                _,
            ) = self.env.step(self.action)
            self.done = terminated or truncated

            observations[step] = self.observation
            actions[step] = self.action
            rewards[step] = self.reward
            next_observations[step] = self.next_observation
            starts[step] = self.start
            dones[step] = self.done
            mask[step] = True
            episode_id[step] = self.episode_id
            self.observation = self.next_observation
            #self.start = self.next_start
            self.start = False

        transitions = {
            "observation": observations, 
            "action": actions, 
            "reward": rewards, 
            "next_observation": next_observations, 
            "start": starts, 
            "done": dones, 
            "mask": mask, 
            "episode_id": episode_id,
        }
        # Last episode will be truncated
        seq_lens.append(self.config['steps_per_epoch'] - sum(seq_lens))
        transitions = self.split_and_pad(transitions, seq_lens)
        self.sampled_frames += step
        self.sampled_epochs += 1
        self.update_episodic_reward(transitions['reward'], transitions['episode_id'])
        episode_reward = self.get_episodic_reward(transitions['episode_id'], transitions["done"])
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        return (
            transitions,
            episode_reward, 
            self.best_reward
        )


class StreamCollector(BatchedSegmentCollector):
    def __call__(self, q_network, policy, progress, key, need_reset=False):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        starts = []
        # Prev dones is what the inference policy sees
        # these are equivalent to starts
        # prev_dones = np.zeros((self.config["segment_length"]), dtype=bool)

        if self.done or need_reset:
            self.episode_reward = 0
            key, reset_key = random.split(key)
            self.done = self.next_done = False
            self.observation, self.start, _ = self.env.reset(
                seed=random.bits(reset_key).item()
            )
            self.recurrent_state = q_network.initial_state()

        for step in range(self.config["segment_length"]):
            if self.sampled_epochs < self.config["random_epochs"]:
                self.action = self.env.action_space.sample()
            else:
                key, action_key = random.split(key)
                self.action, self.recurrent_state = policy(
                    q_network=q_network,
                    x=self.observation,
                    state=self.recurrent_state,
                    start=self.start,
                    done=self.done,
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
                self.next_start,
                _,
            ) = self.env.step(self.action)
            self.done = terminated or truncated

            observations.append(self.observation)
            actions.append(self.action)
            rewards.append(self.reward)
            next_observations.append(self.next_observation)
            starts.append(self.start)
            dones.append(self.done)

            self.observation = self.next_observation
            self.start = self.next_start

            if self.sampled_epochs > self.config["random_epochs"]:
                self.episode_reward += self.reward
            if self.done:
                break

        if not self.config["propagate_state"]:
            self.recurrent_state = q_network.initial_state()

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        starts = np.array(starts)
        dones = np.array(dones)
        mask = np.ones_like(dones)

        self.sampled_frames += step + 1
        self.sampled_epochs += 1
        self.update_best_reward()
        return (
            observations,
            actions,
            rewards,
            next_observations,
            starts,
            dones,
            mask,
            self.get_episodic_reward(),
            self.best_reward
        )
