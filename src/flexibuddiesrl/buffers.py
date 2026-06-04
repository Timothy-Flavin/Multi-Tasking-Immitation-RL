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


def _numpy_to_pinned(src, dtype: th.dtype) -> th.Tensor:
    """Wrap a numpy array as a contiguous CPU tensor and pin it.

    Pinned (page-locked) memory is required for the CUDA DMA engine to perform
    asynchronous H2D transfers via copy_(non_blocking=True).  Without pinning,
    the driver falls back to a synchronous pageable-memory transfer, negating
    any non_blocking benefit.
    """
    t = th.as_tensor(np.ascontiguousarray(src), dtype=dtype)
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
        n_agents: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        action_dtype: th.dtype = th.float32,
        full_gpu: bool = False,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.device = get_device(device)
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.action_dtype = action_dtype
        self.full_gpu = full_gpu

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

    def _init_storage(self, shape, dtype):
        if self.full_gpu:
            return th.zeros(shape, dtype=dtype, device=self.device)
        else:
            return _pin(th.zeros(shape, dtype=dtype))

    @staticmethod
    def swap_and_flatten(tensor: th.Tensor) -> th.Tensor:
        """Flatten the leading [T, E, A] dimensions into a single batch axis.

        Storage layout is [T, E, A, ...]. This collapses the first three dims
        into one, returning [T*E*A, ...].  Because the underlying storage is
        C-contiguous, reshape() returns a view — zero allocation, zero copy.

        Input:  [T, E, A, ...]  or  [T, E, A]  (scalars per step)
        Output: [T*E*A, ...]    or  [T*E*A, 1]
        """
        shape = tensor.shape
        if len(shape) < 4:
            shape = (*shape, 1)
        return tensor.reshape(-1, *shape[3:])

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
        if tensor.device == self.device:
            return tensor
        return tensor.to(self.device, non_blocking=not self.full_gpu)

    def _prep_src(self, src, dtype: th.dtype, shape: tuple) -> th.Tensor:
        """Prepare a source value for a zero-staging copy into a buffer slot.

        - If src is already a Tensor: reshape only; caller owns device placement.
        - If src is numpy / array-like: wrap as contiguous CPU tensor, pin it
          when full_gpu is enabled so that the subsequent copy_(non_blocking=True)
          can hand the transfer to the CUDA DMA engine and return immediately.

        Wrapping a contiguous numpy array with th.as_tensor is zero-copy (the
        tensor shares the array's memory).  Only one real copy happens:
        CPU→pinned (reshape) then DMA into VRAM.
        """
        if isinstance(src, th.Tensor):
            return src.reshape(shape).to(dtype=dtype)
        t = th.as_tensor(np.ascontiguousarray(src), dtype=dtype).reshape(shape)
        if self.full_gpu and th.cuda.is_available():
            t = t.pin_memory()
        return t


class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        n_agents: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        action_dtype: th.dtype = th.float32,
        full_gpu: bool = False,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, n_agents,
            obs_shape=obs_shape, action_dim=action_dim, action_dtype=action_dtype,
            full_gpu=full_gpu
        )
        
        self.buffer_size = max(buffer_size // (n_envs * n_agents), 1)
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination = handle_timeout_termination

        # Initialize storage — layout [T, E, A, O] per CLAUDE.md spec.
        # E before A so that vectorized-env output ([E, A, O]) maps directly to
        # a buffer row with no transpose on insertion.
        self.observations = self._init_storage((self.buffer_size, self.n_envs, self.n_agents, *self.obs_shape), th.float32)
        if not optimize_memory_usage:
            self.next_observations = self._init_storage((self.buffer_size, self.n_envs, self.n_agents, *self.obs_shape), th.float32)

        self.actions = self._init_storage((self.buffer_size, self.n_envs, self.n_agents, self.action_dim), self.action_dtype)
        self.rewards = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.terms = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.truncs = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)

    def add(
        self,
        obs: np.ndarray | th.Tensor,
        next_obs: np.ndarray | th.Tensor,
        action: np.ndarray | th.Tensor,
        reward: np.ndarray | th.Tensor,
        term: np.ndarray | th.Tensor,
        trunc: np.ndarray | th.Tensor,
    ) -> None:
        # Expect inputs to be shaped [n_envs, n_agents, ...] — matches the [E, A, O]
        # order that vectorized environments naturally produce, so no transpose needed.
        # _prep_src wraps numpy inputs as pinned CPU tensors when full_gpu=True
        # so copy_(non_blocking=True) can use the CUDA DMA engine.
        nb = self.full_gpu
        obs_shape  = (self.n_envs, self.n_agents, *self.obs_shape)
        flat_shape = (self.n_envs, self.n_agents)

        self.observations[self.pos].copy_(self._prep_src(obs, th.float32, obs_shape), non_blocking=nb)
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size].copy_(self._prep_src(next_obs, th.float32, obs_shape), non_blocking=nb)
        else:
            self.next_observations[self.pos].copy_(self._prep_src(next_obs, th.float32, obs_shape), non_blocking=nb)

        self.actions[self.pos].copy_(self._prep_src(action, self.action_dtype, (self.n_envs, self.n_agents, self.action_dim)), non_blocking=nb)
        self.rewards[self.pos].copy_(self._prep_src(reward, th.float32, flat_shape), non_blocking=nb)
        self.terms[self.pos].copy_(self._prep_src(term,   th.float32, flat_shape), non_blocking=nb)
        self.truncs[self.pos].copy_(self._prep_src(trunc,  th.float32, flat_shape), non_blocking=nb)

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def _get_samples(self, batch_inds: th.Tensor) -> ReplayBufferSamples:
        # Multi-dim fancy indexing always produces a new pageable CPU tensor —
        # the result loses the pin even though the source was pinned.  Re-pin
        # before to_torch so the async H2D transfer can use DMA.
        need_pin = not self.full_gpu and th.cuda.is_available()

        def _fetch(t: th.Tensor, t_inds, a_inds, e_inds) -> th.Tensor:
            s = t[t_inds, a_inds, e_inds]
            if need_pin:
                s = s.pin_memory()
            return self.to_torch(s)

        # Storage is [T, E, A, ...] — sample E then A for each selected T.
        env_indices   = th.randint(0, self.n_envs,   (len(batch_inds),))
        agent_indices = th.randint(0, self.n_agents, (len(batch_inds),))

        obs     = _fetch(self.observations, batch_inds, env_indices, agent_indices)
        actions = _fetch(self.actions,      batch_inds, env_indices, agent_indices)
        if self.optimize_memory_usage:
            next_obs = _fetch(self.observations, (batch_inds + 1) % self.buffer_size, env_indices, agent_indices)
        else:
            next_obs = _fetch(self.next_observations, batch_inds, env_indices, agent_indices)

        rewards = _fetch(self.rewards, batch_inds, env_indices, agent_indices).reshape(-1, 1)
        terms   = _fetch(self.terms,   batch_inds, env_indices, agent_indices).reshape(-1, 1)
        truncs  = _fetch(self.truncs,  batch_inds, env_indices, agent_indices).reshape(-1, 1)

        return ReplayBufferSamples(obs, actions, next_obs, terms, truncs, rewards)


class RolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        device: th.device | str = "auto",
        n_envs: int = 1,
        n_agents: int = 1,
        obs_shape: tuple[int, ...] = None,
        action_dim: int = None,
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        action_dtype: th.dtype = th.float32,
        log_probs_dim: int = 1,
        full_gpu: bool = False,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, n_agents,
            obs_shape=obs_shape, action_dim=action_dim, action_dtype=action_dtype,
            full_gpu=full_gpu
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.log_probs_dim = log_probs_dim
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        # Layout [T, E, A, ...] — matches env output order, zero-copy insertion.
        self.observations = self._init_storage((self.buffer_size, self.n_envs, self.n_agents, *self.obs_shape), th.float32)
        self.actions = self._init_storage((self.buffer_size, self.n_envs, self.n_agents, self.action_dim), self.action_dtype)
        self.rewards = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.returns = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.terminations = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.truncations = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.values = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.log_probs = self._init_storage((self.buffer_size, self.n_envs, self.n_agents, self.log_probs_dim), th.float32)
        self.advantages = self._init_storage((self.buffer_size, self.n_envs, self.n_agents), th.float32)
        self.final_observations = {} # Key: (pos, agent_idx, env_idx), Val: obs tensor
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, terminations: np.ndarray, truncations: np.ndarray,
        get_value_fn=None,
    ) -> None:
        """Compute GAE advantages and returns in-place.

        Layout violation fixed: the original code did 4 per-step H2D transfers
        (rewards, values, terms, truncs) plus 1 per-step D2H write (advantages)
        inside the reversed loop — 5*T small transfers total.  This version does
        4 bulk H2D transfers before the loop and 1 bulk D2H write after, reducing
        PCIe round-trips from O(T) to O(1) regardless of rollout length.

        For full_gpu=True the .to(device) calls are no-ops (data already on GPU).
        """
        dev = self.device
        T   = self.buffer_size

        # --- One bulk H2D transfer per array (pinned → VRAM via DMA) ---
        # For full_gpu the slices are already on-device; .to() is a no-op.
        rewards = self.rewards.to(dev, non_blocking=True)       # [T, A, E]
        values  = self.values.to(dev, non_blocking=True)        # [T, A, E]
        terms   = self.terminations.to(dev, non_blocking=True)  # [T, A, E]
        truncs  = self.truncations.to(dev, non_blocking=True)   # [T, A, E]

        last_values  = last_values.reshape(self.n_envs, self.n_agents).to(dev)
        last_gae_lam = th.zeros((self.n_envs, self.n_agents), device=dev)

        terms_t  = th.as_tensor(terminations, dtype=th.float32, device=dev).reshape(self.n_envs, self.n_agents)
        truncs_t = th.as_tensor(truncations,  dtype=th.float32, device=dev).reshape(self.n_envs, self.n_agents)

        # Accumulate advantages fully on-device. Layout [T, E, A].
        advantages = th.empty((T, self.n_envs, self.n_agents), device=dev)

        for step in reversed(range(T)):
            if step == T - 1:
                next_non_terminal         = 1.0 - terms_t
                next_episode_continuation = 1.0 - th.clamp(terms_t + truncs_t, 0.0, 1.0)
                next_vals = last_values
            else:
                next_non_terminal         = 1.0 - terms[step]
                next_episode_continuation = 1.0 - th.clamp(terms[step] + truncs[step], 0.0, 1.0)
                next_vals = values[step + 1]

            # Truncation bootstrap: override next_vals for specific (env, agent) pairs.
            # Keys are (t_step, e_idx, a_idx) matching the [T, E, A] storage order.
            # Clone first so we never mutate a view into the bulk `values` tensor.
            if get_value_fn is not None and self.final_observations:
                for (t_step, e_idx, a_idx), final_obs in self.final_observations.items():
                    if t_step == step:
                        next_vals = next_vals.clone()
                        next_vals[e_idx, a_idx] = get_value_fn(final_obs.to(dev))

            delta = rewards[step] + self.gamma * next_vals * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_episode_continuation * last_gae_lam
            advantages[step] = last_gae_lam

        # --- One bulk D2H write-back (VRAM → pinned host via DMA) ---
        # Compute returns on-device to avoid re-transferring values from host.
        if self.full_gpu:
            self.advantages.copy_(advantages)
            self.returns.copy_(advantages + values)
        else:
            adv_cpu = advantages.cpu()          # single D2H transfer
            self.advantages.copy_(adv_cpu)
            # values is still in CPU pinned memory; addition stays on CPU.
            self.returns.copy_(adv_cpu + self.values)

    def add(self, obs, action, reward, termination, truncation, value, log_prob, final_obs=None) -> None:
        # Inputs must be shaped [n_envs, n_agents, ...] — matching [E, A, O] env output.
        # _prep_src wraps numpy inputs as pinned CPU tensors when full_gpu=True so
        # copy_(non_blocking=True) activates the CUDA DMA engine for each field.
        nb         = self.full_gpu
        flat_shape = (self.n_envs, self.n_agents)

        self.observations[self.pos].copy_(self._prep_src(obs, th.float32, (self.n_envs, self.n_agents, *self.obs_shape)), non_blocking=nb)
        self.actions[self.pos].copy_(self._prep_src(action, self.action_dtype, (self.n_envs, self.n_agents, self.action_dim)), non_blocking=nb)
        self.rewards[self.pos].copy_(self._prep_src(reward, th.float32, flat_shape), non_blocking=nb)
        self.terminations[self.pos].copy_(self._prep_src(termination, th.float32, flat_shape), non_blocking=nb)
        self.truncations[self.pos].copy_(self._prep_src(truncation,  th.float32, flat_shape), non_blocking=nb)
        self.values[self.pos].copy_(self._prep_src(value, th.float32, flat_shape), non_blocking=nb)
        self.log_probs[self.pos].copy_(self._prep_src(log_prob, th.float32, (self.n_envs, self.n_agents, self.log_probs_dim)), non_blocking=nb)
        
        if final_obs is not None:
            if isinstance(final_obs, dict):
                for (e_idx, a_idx), f_obs in final_obs.items():
                    self.final_observations[(self.pos, e_idx, a_idx)] = th.as_tensor(f_obs).clone()
            else:
                # truncation broadcast to [E, A]; walk in [E, A] order
                t_np = truncation.cpu().numpy() if th.is_tensor(truncation) else np.array(truncation)
                trunc_indices = np.where(t_np.reshape(self.n_envs, self.n_agents))
                for e_idx, a_idx in zip(*trunc_indices):
                    self.final_observations[(self.pos, e_idx, a_idx)] = th.as_tensor(final_obs[e_idx, a_idx]).clone()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_agents * self.n_envs)

        observations = self.swap_and_flatten(self.observations)
        actions = self.swap_and_flatten(self.actions)
        values = self.swap_and_flatten(self.values)
        log_probs = self.swap_and_flatten(self.log_probs)
        advantages = self.swap_and_flatten(self.advantages)
        returns = self.swap_and_flatten(self.returns)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_agents * self.n_envs:
            end_idx = start_idx + (batch_size if batch_size is not None else self.buffer_size * self.n_agents * self.n_envs)
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
        # Fancy indexing (t[batch_inds]) allocates a new pageable CPU tensor —
        # the result is no longer pinned even though the source was.  Re-pin
        # so that to_torch(non_blocking=True) can hand the transfer to the
        # CUDA DMA engine and return immediately instead of blocking.
        need_pin = not self.full_gpu and th.cuda.is_available()

        def _fetch(t: th.Tensor) -> th.Tensor:
            s = t[batch_inds]
            if need_pin:
                s = s.pin_memory()
            return self.to_torch(s)

        return RolloutBufferSamples(
            observations=_fetch(observations),
            actions=_fetch(actions),
            old_values=_fetch(values),
            old_log_prob=_fetch(log_probs),
            advantages=_fetch(advantages),
            returns=_fetch(returns),
        )
