from tape_collector import TapeCollector
import unittest
import gymnasium as gym
import numpy as np
import jax

class FakeEnv(gym.Env):
    dones   = [0, 0, 0, 1, 0, 1, 0, 0, 1]
    rewards = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    obs =     [0, 0, 0, 0, 0, 0, 0, 0, 0]
    reset_obs = [1, 1, 1]
    observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.i = 0
        self.num_resets = 0

    def reset(self, *args, **kwargs):
        out = self.reset_obs[self.num_resets]
        self.num_resets += 1
        return out, {}

    def step(self, action, *args):
        obs = self.obs[self.i]
        reward = self.rewards[self.i]
        done = self.dones[self.i]
        self.i += 1
        return obs, reward, done, done, {}

def fake_policy(*args, **kwargs):
    return np.array(0), np.array(0)

class FakeQNet:
    s = 0
    def initial_state(self):
        self.s += 1
        return np.array(self.s)

class TestTapeCollector(unittest.TestCase):
    def test_seq(self):
        env = FakeEnv()
        config = {"collect": {"steps_per_epoch": 3, "random_epochs": 0, "eps_start": 0, "eps_end": 0}}
        collector = TapeCollector(env, config)
        for i in range(3):
            trans, reward, best_reward = collector(
                q_network=FakeQNet(),
                policy=fake_policy,
                progress=0,
                key=jax.random.PRNGKey(0),
                )
        
if __name__ == '__main__':
    unittest.main()