import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset
from utils.occlusions import make_occlusion_module


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


class OcclusionWrapper(gymnasium.Wrapper):
    """Environment wrapper to occlude observations."""
    def __init__(self, env, occlusion_config):
        super().__init__(env)
        self.occluder = make_occlusion_module(len(env.observation_space.shape), occlusion_config)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        return self.occluder(ob), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, config):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        config: config

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if config['frame_stack'] is not None:
        env = FrameStackWrapper(env, config['frame_stack'])

    if config.get('occlusion') is not None:
        env = OcclusionWrapper(env, config['occlusion'])

    env.reset()

    return env, train_dataset, val_dataset
