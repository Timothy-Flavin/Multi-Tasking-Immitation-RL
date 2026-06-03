# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

try:
    import psutil
except ImportError:
    psutil = None

__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    terminations: th.Tensor
    truncations: th.Tensor
    rewards: th.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def _pin(t: th.Tensor) -> th.Tensor:
    """Pin tensor to page-locked memory only when a CUDA device is available."""
    return t.pin_memory() if th.cuda.is_available() else t


def get_device(device: th.device | str = "auto") -> th.device:
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    device = th.device(device)
    if device.type == "cuda" and not th.cuda.is_available():
        return th.device("cpu")
    return device

def supports_int32_gather():
    x = th.randn(2, 3, device='cpu')
    idx = th.tensor([[0, 1, 2], [2, 1, 0]], dtype=th.int32, device='cpu')
    try:
        th.gather(x, 1, idx)
    except RuntimeError:
        print("Torch does not support int32 gather, switching to 64")
        return False
        
    if th.cuda.is_available():
        x = x.to('cuda')
        idx = idx.to('cuda')
        try:
            th.gather(x, 1, idx)
        except RuntimeError:
            print("Torch does not support int32 gather on cuda, switching to 64")
            return False
            
    return True

class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        action_dtype: th.dtype = th.float32,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.device = get_device(device)
        self.n_envs = n_envs
        self.action_dtype = action_dtype
        # 1. Auto-infer integer dtype if the action space is discrete/binary
        if action_space is not None and isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
            if self.action_dtype == th.float32: # Override the default float32
                self.action_dtype = th.int32

        # 2. Apply your fallback with the correct 'th' namespace
        if self.action_dtype == th.int32:
            if not supports_int32_gather():
                self.action_dtype = th.int64

        # Use custom dim if provided, else parse from space
        if obs_shape is not None:
            self.obs_shape = obs_shape
        else:
            self.obs_shape = get_obs_shape(observation_space)

        if action_dim is not None:
            self.action_dim = action_dim
        else:
            self.action_dim = get_action_dim(action_space)

        self.pos = 0
        self.full = False

    @staticmethod
    def swap_and_flatten(tensor: th.Tensor) -> th.Tensor:
        shape = tensor.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return tensor.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = th.randint(0, upper_bound, (batch_size,))
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples | RolloutBufferSamples:
        raise NotImplementedError()

    def to_torch(self, tensor: th.Tensor) -> th.Tensor:
        return tensor.to(self.device, non_blocking=True)


class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        action_dtype: th.dtype = th.float32,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, 
            obs_shape=obs_shape, action_dim=action_dim, action_dtype=action_dtype
        )
        
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination = handle_timeout_termination

        # Initialize Pinned Tensors
        self.observations = _pin(th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32))
        if not optimize_memory_usage:
            self.next_observations = _pin(th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32))

        self.actions = _pin(th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_dtype))
        self.rewards = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.terms = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.truncs = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))

    def add(
        self,
        obs: np.ndarray | th.Tensor,
        next_obs: np.ndarray | th.Tensor,
        action: np.ndarray | th.Tensor,
        reward: np.ndarray | th.Tensor,
        term: np.ndarray | th.Tensor,
        trunc: np.ndarray | th.Tensor,
    ) -> None:
        self.observations[self.pos].copy_(th.as_tensor(obs).reshape((self.n_envs, *self.obs_shape)))
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size].copy_(th.as_tensor(next_obs).reshape((self.n_envs, *self.obs_shape)))
        else:
            self.next_observations[self.pos].copy_(th.as_tensor(next_obs).reshape((self.n_envs, *self.obs_shape)))

        self.actions[self.pos].copy_(th.as_tensor(action).reshape((self.n_envs, self.action_dim)))
        self.rewards[self.pos].copy_(th.as_tensor(reward))
        self.terms[self.pos].copy_(th.as_tensor(term))
        self.truncs[self.pos].copy_(th.as_tensor(trunc))

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples:
        env_indices = th.randint(0, self.n_envs, (len(batch_inds),))
        obs = self.observations[batch_inds, env_indices]
        actions = self.actions[batch_inds, env_indices]
        
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices]
        else:
            next_obs = self.next_observations[batch_inds, env_indices]

        rewards = self.rewards[batch_inds, env_indices].reshape(-1, 1)
        terms = self.terms[batch_inds, env_indices].reshape(-1, 1)
        truncs = self.truncs[batch_inds, env_indices].reshape(-1, 1)

        return ReplayBufferSamples(
            self.to_torch(obs), self.to_torch(actions), self.to_torch(next_obs),
            self.to_torch(terms), self.to_torch(truncs), self.to_torch(rewards)
        )


class RolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        action_dtype: th.dtype = th.float32,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs,
            obs_shape=obs_shape, action_dim=action_dim, action_dtype=action_dtype
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = _pin(th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32))
        self.actions = _pin(th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_dtype))
        self.rewards = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.returns = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.terminations = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.truncations = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.values = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.log_probs = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.advantages = _pin(th.zeros((self.buffer_size, self.n_envs), dtype=th.float32))
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, terminations: np.ndarray, truncations: np.ndarray
    ) -> None:
        last_values = last_values.flatten()
        last_gae_lam = 0
        
        terms_t = th.as_tensor(terminations, dtype=th.float32).flatten()
        truncs_t = th.as_tensor(truncations, dtype=th.float32).flatten()

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # Value bootstrap applies if episode truncates, but not if it terminates natively.
                next_non_terminal = 1.0 - terms_t
                
                # Prevent GAE advantage from bleeding over an episode boundary (either terminated or truncated)
                next_episode_continuation = 1.0 - th.clamp(terms_t + truncs_t, 0.0, 1.0)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.terminations[step]
                next_episode_continuation = 1.0 - th.clamp(self.terminations[step] + self.truncations[step], 0.0, 1.0)
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_episode_continuation * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns.copy_(self.advantages + self.values)

    def add(self, obs, action, reward, termination, truncation, value, log_prob) -> None:
        self.observations[self.pos].copy_(th.as_tensor(obs).reshape((self.n_envs, *self.obs_shape)))
        self.actions[self.pos].copy_(th.as_tensor(action).reshape((self.n_envs, self.action_dim)))
        self.rewards[self.pos].copy_(th.as_tensor(reward))
        self.terminations[self.pos].copy_(th.as_tensor(termination))
        self.truncations[self.pos].copy_(th.as_tensor(truncation))
        self.values[self.pos].copy_(value.flatten())
        self.log_probs[self.pos].copy_(log_prob.reshape(self.n_envs))
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data by swapping and flattening (Time, Envs) -> (Time * Envs)
        # We use the staticmethod swap_and_flatten from BaseBuffer
        observations = self.swap_and_flatten(self.observations)
        actions = self.swap_and_flatten(self.actions)
        values = self.swap_and_flatten(self.values)
        log_probs = self.swap_and_flatten(self.log_probs)
        advantages = self.swap_and_flatten(self.advantages)
        returns = self.swap_and_flatten(self.returns)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            # If batch_size is None, return all data at once
            end_idx = start_idx + (batch_size if batch_size is not None else self.buffer_size * self.n_envs)
            yield self._get_samples(indices[start_idx:end_idx], 
                                    observations, actions, values, 
                                    log_probs, advantages, returns)
            start_idx = end_idx

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        observations: th.Tensor,
        actions: th.Tensor,
        values: th.Tensor,
        log_probs: th.Tensor,
        advantages: th.Tensor,
        returns: th.Tensor,
    ) -> RolloutBufferSamples:
        return RolloutBufferSamples(
            observations=self.to_torch(observations[batch_inds]),
            actions=self.to_torch(actions[batch_inds]),
            old_values=self.to_torch(values[batch_inds]),
            old_log_prob=self.to_torch(log_probs[batch_inds]),
            advantages=self.to_torch(advantages[batch_inds]),
            returns=self.to_torch(returns[batch_inds]),
        )