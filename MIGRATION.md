# Migration Guide: Legacy → New Buffer Implementations

This guide covers migrating from the single-environment legacy agents
(`DQN`, `SAC`, `PG_stabalized`) to the vectorized new-buffer agents
(`DQN_new_buffer`, `SAC_new_buffer`, `PG_new_buffer`).

The new implementations replace the `flexibuff` dependency with a
built-in `buffers` module, support `[T, A, E, *]`-layout tensors for
parallel multi-environment training, and are compatible with EnvPool
and similar vectorized environment runners.

---

## Table of Contents

1. [Why Migrate](#why-migrate)
2. [Buffer API](#buffer-api)
   - [ReplayBuffer (off-policy)](#replaybuffer-off-policy)
   - [RolloutBuffer (on-policy)](#rolloutbuffer-on-policy)
3. [DQN](#dqn)
4. [SAC](#sac)
5. [PG / PPO](#pg--ppo)
6. [Full Training Loop Examples](#full-training-loop-examples)

---

## Why Migrate

| Feature | Legacy | New |
|---|---|---|
| Observation layout | `[B, obs_dim]` flat | `[T, A, E, obs_dim]` + auto-flatten |
| Buffer dependency | `flexibuff.FlexibleBuffer` | Built-in `ReplayBuffer` / `RolloutBuffer` |
| Parallel envs | Single env only | N envs, N agents natively |
| GPU data path | CPU→GPU per call | Pinned memory + async DMA (full_gpu) |
| Truncation bootstrap | Manual | Built-in via `final_observations` |
| `nn.Module` base | `SAC`, `PG` only | All three agents |

---

## Buffer API

### ReplayBuffer (off-policy)

Used by **DQN** and **SAC**.

```python
from flexibuddiesrl.buffers import ReplayBuffer

buffer = ReplayBuffer(
    buffer_size=1_000_000,   # total transitions (divided by n_agents*n_envs internally)
    obs_shape=(obs_dim,),    # tuple, e.g. (4,) for CartPole
    action_dim=1,            # total number of action scalars stored per agent per env
                             #   discrete-only: len(discrete_action_dims)
                             #   continuous-only: continuous_action_dim
                             #   mixed: len(discrete_action_dims) + continuous_action_dim
    device="cuda",
    n_envs=64,               # number of parallel environments
    n_agents=1,              # number of agents (parameter-shared)
    action_dtype=torch.float32,  # use torch.int32/int64 for pure-discrete actions
    full_gpu=False,          # True = keep all data on GPU (faster on large buffers)
)
```

**Adding transitions** — inputs must be shaped `[n_envs, n_agents, ...]`,
matching the `[E, A, O]` order that vectorized environments naturally produce.
For single-agent scenarios add the agent dimension in position 1:

```python
# obs from envpool/gym-vec is [n_envs, obs_dim]
buffer.add(
    obs=obs[:, np.newaxis, :],           # [64, 1, obs_dim]
    next_obs=next_obs[:, np.newaxis, :], # [64, 1, obs_dim]
    action=actions,                      # [64, 1, action_dim]
    reward=rewards[:, np.newaxis],       # [64, 1]
    term=terminations[:, np.newaxis],    # [64, 1]
    trunc=truncations[:, np.newaxis],    # [64, 1]
)
```

**Sampling** returns a `ReplayBufferSamples` namedtuple:

```python
samples = buffer.sample(batch_size=256)
# samples.observations      [B, obs_dim]
# samples.actions           [B, action_dim]
# samples.next_observations [B, obs_dim]
# samples.terminations      [B, 1]
# samples.truncations       [B, 1]
# samples.rewards           [B, 1]
```

The sampler randomly picks across the time, agent, and environment
dimensions, so `B` samples come from a mix of all stored transitions.

---

### RolloutBuffer (on-policy)

Used by **PG/PPO**.

```python
from flexibuddiesrl.buffers import RolloutBuffer

collection_steps = 2048 // n_envs   # steps to collect before each update

buffer = RolloutBuffer(
    buffer_size=collection_steps,
    obs_shape=(obs_dim,),
    action_dim=1,               # same convention as ReplayBuffer
    n_envs=64,
    n_agents=1,
    device="cuda",
    log_probs_dim=1,            # 1 for discrete; continuous_action_dim for continuous
    action_dtype=torch.float32,
    full_gpu=False,
)
```

**Adding transitions** — called once per parallel environment step:

```python
buffer.add(
    obs=obs[np.newaxis, ...],            # [1, n_envs, obs_dim]
    action=actions,                      # [1, n_envs, action_dim]
    reward=rewards[np.newaxis, ...],     # [1, n_envs]
    termination=terms[np.newaxis, ...],  # [1, n_envs]
    truncation=truncs[np.newaxis, ...],  # [1, n_envs]
    value=values,                        # tensor [1, n_envs] or [n_agents, n_envs]
    log_prob=log_probs,                  # tensor [1, n_envs, log_probs_dim]
    final_obs=final_obs,                 # optional [1, n_envs, obs_dim] for truncations
)
```

After the rollout is collected, call `reinforcement_learn` (which
internally computes GAE) then reset:

```python
last_values = agent.expected_V(torch.as_tensor(obs, device=device).unsqueeze(0))
agent.reinforcement_learn(buffer, last_values, terminations, truncations)
buffer.reset()
```

The buffer also exposes a `get(batch_size)` generator for manual
mini-batch iteration if you call `compute_returns_and_advantage`
yourself first.

**Key layout rule:** All tensors stored in both buffers follow the
`[T, A, E, *]` convention (time × agents × envs × feature), which
maps directly to how the new agents consume them.

---

## DQN

### Import change

```python
# Legacy
from flexibuddiesrl.DQN import DQN

# New
from flexibuddiesrl.DQN_new_buffer import DQN
```

### Constructor changes

| Parameter | Legacy | New | Notes |
|---|---|---|---|
| `init_eps` | ✓ | **Removed** | Fixed epsilon passed at call time instead |
| `eps_decay_half_life` | ✓ | **Removed** | |
| `conservative` | ✓ | **Removed** | CQL removed from both implementations |
| `tau` | hardcoded 0.005 | `tau=0.005` | Now a constructor param |
| `target_update_interval` | every step | `target_update_interval=1` | Now a constructor param |
| `batch_size` | external | `batch_size=256` | Stored but not used internally |
| `entropy` | ✓ | ✓ (same) | Enables Soft-DQN |
| `munchausen` | ✓ | ✓ (same) | Enables Munchausen-DQN |
| `imitation_lr` | ✓ | ✓ (same) | Separate optimizer for imitation steps |
| `head_hidden_dims` | `head_hidden_dim` (single) | `head_hidden_dims` (list) | API unified |

```python
# Legacy
agent = DQN(
    obs_dim=4,
    discrete_action_dims=[2],
    init_eps=0.9,
    eps_decay_half_life=10000,
)

# New
agent = DQN(
    obs_dim=4,
    discrete_action_dims=[2],
    tau=0.005,
    target_update_interval=1,
)
```

### `train_actions` changes

Epsilon is now a **per-call parameter** (no more internal decay state).
The legacy `train_actions` also now accepts `epsilon`; both interfaces
are equivalent.

```python
# Legacy — epsilon was internal state, decayed automatically
act = agent.train_actions(obs, step=True)

# New — pass epsilon explicitly (no decay; implement your own schedule
# externally if needed)
act = agent.train_actions(obs, epsilon=0.1)    # greedy with 10% random
act = agent.train_actions(obs, epsilon=0.0)    # fully greedy (eval)
```

The new `train_actions` also accepts **3D observations** directly:

```python
# Single env — shape [obs_dim]  or  [B, obs_dim]
act = agent.train_actions(obs)

# Multi-env — shape [n_envs, n_agents, obs_dim] per [T,E,A,O] spec; results are also [E,A,...]
obs_t = torch.as_tensor(obs).unsqueeze(1)   # [n_envs, 1, obs_dim]  ← agent dim at position 1
act = agent.train_actions(obs_t)
# act["discrete_actions"]  shape: [n_envs, 1, n_heads]
# act["continuous_actions"] shape: [n_envs, 1, n_cont_dims] or None
# act["values"]             shape: [n_envs, 1]
```

### `reinforcement_learn` changes

The new DQN accepts `ReplayBufferSamples` directly (not `FlexiBatch`).

```python
# Legacy
mb = buffer.sample_transitions(batch_size=256, as_torch=True, device=device)
agent.reinforcement_learn(mb, agent_num=0)

# New
samples = replay_buffer.sample(batch_size=256)
agent.reinforcement_learn(samples)
# Returns: {"rl_loss": float, "importance_raw": ndarray (QMIX only)}
```

### `imitation_learn` changes

The legacy returned `{"im_discrete_loss": ..., "im_continuous_loss": ...}`
and used `self.optimizer` for updates. The new implementation is
functionally identical but uses a **separate `imitation_optimizer`**
(Adam with `imitation_lr`) so imitation steps don't corrupt the RL
optimizer's momentum.

```python
result = agent.imitation_learn(
    observations=obs_batch,         # [B, obs_dim]
    continuous_actions=c_actions,   # [B, c_dim] or None
    discrete_actions=d_actions,     # [B, n_heads] long or None
)
# {"im_discrete_loss": float, "im_continuous_loss": float}
```

---

## SAC

### Import change

```python
# Legacy
from flexibuddiesrl.SAC import SAC

# New
from flexibuddiesrl.SAC_new_buffer import SAC
```

### Constructor changes

| Parameter | Legacy default | New default | Notes |
|---|---|---|---|
| `sac_tau` | `0.05` | `0.05` | Matches legacy; **do not change to 0.005** for multi-env training — target-tracking stability degrades significantly (see note below) |
| `hidden_dims` | `[32, 32]` | `[256, 256]` | Override explicitly for comparisons |

**`sac_tau` warning:** The SAC paper uses `tau=0.005` for single-env
training with 1 gradient step per env step. With N parallel envs and
N/update_every gradient steps per parallel step, the target can become
very stale. The legacy default of `0.05` is required for stable
convergence in batched training. Only use `0.005` if running a single
environment with 1 gradient step per transition.

The new SAC is now an `nn.Module` in addition to `Agent`:

```python
# Legacy: Agent only — cannot call state_dict() natively
# New: nn.Module + Agent — supports PyTorch module introspection

# Check parameter count
actor_params, total_params = agent.param_count()

# Move to device
agent.to("cuda")
```

### `train_actions` changes

Input format and returns are the same. The new implementation also
supports **3D inputs** (`[n_agents, n_envs, obs_dim]`):

```python
obs_t = torch.as_tensor(obs).unsqueeze(0)   # [1, n_envs, obs_dim]
act = agent.train_actions(obs_t)
# act["discrete_actions"]  shape: [1, n_envs, n_heads] or None
# act["continuous_actions"] shape: [1, n_envs, c_dim] or None
```

### `reinforcement_learn` changes

Accepts `ReplayBufferSamples` instead of `FlexiBatch`.

```python
# Legacy
mb = buffer.sample_transitions(256, as_torch=True, device=device)
agent.reinforcement_learn(mb, agent_num=0)

# New
samples = replay_buffer.sample(256)
agent.reinforcement_learn(samples)
# Returns: {"critic_loss": float, "actor_loss": float (if actor updated),
#           "alpha_c": float, "alpha_d": float}
```

### `save` / `load` changes

The legacy `save` stored only the network weights in a custom dict;
`load` required manually reconstructing the model first.

The new implementation stores a **complete checkpoint**: all network
weights (including target networks), all optimizer states, and the
temperature parameters. The checkpoint is self-contained.

```python
# Save
agent.save("checkpoints/sac_run1.pt")

# Load — construct agent first with the same hyperparams, then load
agent = SAC(obs_dim=..., ...)
agent.load("checkpoints/sac_run1.pt")
```

The checkpoint dict keys are:
`actor`, `Q1`, `Q1_target`, `Q2`, `Q2_target`,
`actor_opt`, `Q1_opt`, `Q2_opt`,
`alpha_opt_c`, `alpha_opt_d`, `log_alpha_c`, `log_alpha_d`.

---

## PG / PPO

### Import change

```python
# Legacy
from flexibuddiesrl.PG_stabalized import PG

# New
from flexibuddiesrl.PG_new_buffer import PG
```

### Constructor changes

All constructor parameters are preserved. New additions:

| Parameter | Default | Notes |
|---|---|---|
| `use_kl_penalty` | `False` | Enables KL divergence penalty in QMIX/VDN mode |
| `offline_critic_buffer` | `False` | Enables replay buffer for critic stabilization in QMIX/VDN |
| `wall_time` | `False` | Add `rl_time` / `act_time` to result dicts |

The `print(f"PPO.py Continuous action dim: ...")` debug line present in
the legacy `__init__` has been removed.

### `train_actions` changes

The new `train_actions` additionally returns `"values"` — the critic's
current value estimate for each observation. This is required to
pre-fill the `RolloutBuffer` efficiently (avoids a second forward pass
during GAE computation).

```python
act = agent.train_actions(obs_t)
# act["discrete_actions"]       shape: [1, n_envs, n_heads] or None
# act["continuous_actions"]     shape: [1, n_envs, c_dim] or None
# act["discrete_log_probs"]     shape: [1, n_envs, n_heads] or None
# act["continuous_log_probs"]   shape: [1, n_envs, c_dim] or None  -- NOTE: NOT summed
# act["values"]                 shape: [1, n_envs]   ← NEW
# act["act_time"]               float (0 unless wall_time=True)
```

**Log-prob layout note:** `continuous_log_probs` is per-dimension
`[B, c_dim]`, not a scalar. Sum over the last dimension yourself if
you need a joint log-prob for ratio-based clipping:

```python
joint_lp = act["continuous_log_probs"].sum(-1)   # [1, n_envs]
```

### `reinforcement_learn` changes

The new signature replaces `FlexiBatch` with a `RolloutBuffer` plus
explicit bootstrap tensors:

```python
# Legacy
agent.reinforcement_learn(batch, agent_num=0, critic_only=False)

# New
agent.reinforcement_learn(
    buffer,             # RolloutBuffer — must be full (buffer.full == True)
    last_values,        # torch.Tensor [n_agents, n_envs] — V(s_T) for GAE bootstrap
    last_terminations,  # np.ndarray [n_agents, n_envs] — whether s_T was terminal
    last_truncations,   # np.ndarray [n_agents, n_envs] — whether s_T was truncated
    critic_only=False,  # QMIX/VDN only — skip actor update
    debug=False,
)
```

The `agent_num` parameter is no longer needed: the buffer already
contains all agent data in `[T, A, E, *]` layout.

**Return dict additions** (QMIX/VDN mode only):

```python
result = agent.reinforcement_learn(...)
# result["rl_actor_loss"]      float
# result["rl_critic_loss"]     float
# result["joint_kl"]           float   ← NEW (KL between old/new distributions)
# result["importance_per_dim"] ndarray ← NEW (mean credit-assignment weights per action)
# result["importance_raw"]     ndarray ← NEW (per-step credit-assignment weights)
```

### Value clipping

The new implementation respects the `value_clip` constructor parameter
in the standard (non-QMIX) path. Pass it at construction time:

```python
agent = PG(obs_dim=4, ..., value_clip=0.5)   # PPO-style value clipping
agent = PG(obs_dim=4, ..., value_clip=0.0)   # disabled (default)
```

---

## Full Training Loop Examples

### Off-policy (DQN / SAC) with 64 parallel envs

```python
import numpy as np
import torch
import envpool
from flexibuddiesrl.DQN_new_buffer import DQN       # or SAC_new_buffer.SAC
from flexibuddiesrl.buffers import ReplayBuffer

N_ENVS      = 64
OBS_DIM     = 4
ACTION_DIM  = 1          # one discrete head
BATCH_SIZE  = 256
UPDATE_EVERY = 8         # 1 gradient step per 8 env transitions (legacy ratio)
TOTAL_STEPS = 200_000
WARMUP      = 1000       # start training after this many env transitions

env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=N_ENVS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQN(
    obs_dim=OBS_DIM,
    discrete_action_dims=[2],
    hidden_dims=[64, 64],
    device=device,
    lr=1e-3,
)

buffer = ReplayBuffer(
    buffer_size=1_000_000,
    obs_shape=(OBS_DIM,),
    action_dim=ACTION_DIM,
    n_envs=N_ENVS,
    n_agents=1,
    device=device,
)

# Number of gradient steps per parallel step keeps update/env-step ratio constant
# Legacy: 1 update per 8 single-env steps
# New:    N_ENVS / UPDATE_EVERY = 8 updates per parallel step
num_updates_per_step = N_ENVS // UPDATE_EVERY

obs = env.reset()[0]
steps_done = 0

while steps_done < TOTAL_STEPS:
    obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)  # [1, N_ENVS, obs_dim]
    with torch.no_grad():
        act = agent.train_actions(obs_t, epsilon=0.1)
    raw_actions = act["discrete_actions"].flatten()           # [N_ENVS]

    obs_next, rewards, terms, truncs, _ = env.step(raw_actions)

    buffer.add(
        obs=obs[np.newaxis],           # [1, N_ENVS, obs_dim]
        next_obs=obs_next[np.newaxis],
        action=act["discrete_actions"],  # [1, N_ENVS, 1]
        reward=rewards[np.newaxis],
        term=terms[np.newaxis],
        trunc=truncs[np.newaxis],
    )
    obs = obs_next
    steps_done += N_ENVS

    if steps_done >= WARMUP:
        for _ in range(num_updates_per_step):
            samples = buffer.sample(BATCH_SIZE)
            agent.reinforcement_learn(samples)
```

---

### On-policy (PPO) with 64 parallel envs

```python
import numpy as np
import torch
import envpool
from flexibuddiesrl.PG_new_buffer import PG
from flexibuddiesrl.buffers import RolloutBuffer

N_ENVS          = 64
OBS_DIM         = 4
ROLLOUT_STEPS   = 2048 // N_ENVS    # parallel steps per rollout (= 32)
TOTAL_STEPS     = 200_000

env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=N_ENVS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = PG(
    obs_dim=OBS_DIM,
    discrete_action_dims=[2],
    hidden_dims=[64, 64],
    device=device,
    n_epochs=4,
    mini_batch_size=128,
    lr=3e-4,
)

buffer = RolloutBuffer(
    buffer_size=ROLLOUT_STEPS,
    obs_shape=(OBS_DIM,),
    action_dim=1,           # one discrete head
    n_envs=N_ENVS,
    n_agents=1,
    device=device,
    log_probs_dim=1,        # one log-prob per discrete head
    action_dtype=torch.int32,
)

obs = env.reset()[0]
steps_done = 0

while steps_done < TOTAL_STEPS:
    # --- Collect rollout ---
    agent.eval()
    for _ in range(ROLLOUT_STEPS):
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)  # [1, N_ENVS, obs_dim]
        with torch.no_grad():
            act = agent.train_actions(obs_t)

        raw_actions = act["discrete_actions"].flatten()
        obs_next, rewards, terms, truncs, info = env.step(raw_actions)

        # Collect final observations for truncated episodes (bootstrap obs)
        final_obs = None
        if np.any(truncs):
            final_obs = np.zeros_like(obs)[np.newaxis]
            src = info.get("final_observation") or info.get("obs")
            if src is not None:
                for i in range(N_ENVS):
                    if truncs[i]:
                        final_obs[0, i] = src[i]

        buffer.add(
            obs=obs[np.newaxis],
            action=act["discrete_actions"],          # [1, N_ENVS, 1]
            reward=rewards[np.newaxis],
            termination=terms[np.newaxis],
            truncation=truncs[np.newaxis],
            value=torch.as_tensor(act["values"]),    # [1, N_ENVS]
            log_prob=torch.as_tensor(act["discrete_log_probs"]),  # [1, N_ENVS, 1]
            final_obs=final_obs,
        )
        obs = obs_next
        steps_done += N_ENVS

    # --- Update ---
    agent.train()
    obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
    with torch.no_grad():
        last_values = agent.expected_V(obs_t)   # [1, N_ENVS]

    result = agent.reinforcement_learn(buffer, last_values, terms[np.newaxis], truncs[np.newaxis])
    buffer.reset()

    print(f"Step {steps_done}: actor={result['rl_actor_loss']:.4f} critic={result['rl_critic_loss']:.4f}")
```

---

## Quick Reference: Return Dict Changes

| Agent | Key | Legacy | New |
|---|---|---|---|
| DQN | `rl_loss` | ✓ | ✓ |
| DQN | `importance_raw` | — | ✓ (QMIX only) |
| DQN | `act_time` | ✓ | — |
| SAC | `critic_loss` | ✓ | ✓ |
| SAC | `actor_loss` | ✓ | ✓ |
| SAC | `alpha_c`, `alpha_d` | ✓ | ✓ |
| PG | `rl_actor_loss`, `rl_critic_loss` | ✓ | ✓ |
| PG | `values` in `train_actions` | — | ✓ |
| PG | `joint_kl` | ✓ (QMIX) | ✓ (QMIX) |
| PG | `importance_per_dim` | ✓ (QMIX) | ✓ (QMIX) |
| PG | `importance_raw` | ✓ (QMIX) | ✓ (QMIX) |
