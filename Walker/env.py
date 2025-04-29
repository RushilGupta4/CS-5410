# env.py

import gymnasium as gym
from utils import RunningMeanStd


def make_env(env_name: str, seed: int, hardcore: bool):
    """
    Create a seeded Gym environment plus its observation normalizer.
    """
    env = gym.make(env_name, hardcore=hardcore)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    return env, obs_rms
