"""
LunarLanderContinuous-v2 benchmark via envpool.

Hybrid action space: one discrete head (5 levels → [-1, -0.5, 0, 0.5, 1]) for the
main engine, plus one continuous action for the lateral thruster.

Steps: SAC / DQN = 300 k,  PPO = 600 k.
Results written to lander_results/.
"""

import envpool
import numpy as np
import torch as th
import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from flexibuddiesrl.PG_new_buffer import PG
from flexibuddiesrl.SAC_new_buffer import SAC
from flexibuddiesrl.DQN_new_buffer import DQN
from flexibuddiesrl.buffers import RolloutBuffer, ReplayBuffer

# --- Env & action-space constants -------------------------------------------

# Discrete main-engine lookup: index → thrust value
ACTION_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
N_DISC = len(ACTION_LEVELS)  # 5

OBS_DIM = 8
DISC_DIMS = [N_DISC]  # one discrete head
CONT_DIM = 1  # lateral thruster [-1, 1]
MAX_ACTIONS = np.array([1.0], dtype=np.float32)
MIN_ACTIONS = np.array([-1.0], dtype=np.float32)

# Buffer action layout: col 0 = disc index (float), col 1 = cont value
ACTION_DIM = len(DISC_DIMS) + CONT_DIM  # 2

# --- Training constants ------------------------------------------------------

NUM_ENVS = 64
BATCH_SIZE = 512
LR = 3e-4
SEEDS = [0, 1, 2]
STEPS_OFF_POLICY = 300_000  # SAC, DQN
STEPS_ON_POLICY = 600_000  # PPO

# Updates per batch step scaled to maintain the same update/transition ratio as
# the 16-env toy runner (1 update per batched step × 16 envs = 1/16 per transition).
# With 64 envs: 4 updates per batched step → same 1/16 ratio.
UPDATES_PER_STEP = NUM_ENVS // 8

CONFIGS = {
    "PPO_shared_nomix": dict(algo="PPO", mix_type=None),
    "PPO_VDN": dict(algo="PPO", mix_type="VDN"),
    "PPO_QMIX": dict(algo="PPO", mix_type="QMIX", mixer_dim=64),
    "PPO_VDN_offline": dict(algo="PPO", mix_type="VDN", offline_critic=True),
    "PPO_QMIX_offline": dict(algo="PPO", mix_type="QMIX", mixer_dim=64, offline_critic=True),
    "SAC_Q": dict(algo="SAC", mode="Q"),
    "SAC_V": dict(algo="SAC", mode="V"),
    "DQN_shared_nomix": dict(algo="DQN", mix_type=None),
    "DQN_VDN": dict(algo="DQN", mix_type="VDN"),
    "DQN_QMIX": dict(algo="DQN", mix_type="QMIX", mixer_dim=64),
}

# ---------------------------------------------------------------------------


def _make_model(cfg, device):
    algo = cfg["algo"]
    if algo == "PPO":
        return PG(
            obs_dim=OBS_DIM,
            continuous_action_dim=CONT_DIM,
            discrete_action_dims=DISC_DIMS,
            max_actions=MAX_ACTIONS,
            min_actions=MIN_ACTIONS,
            device=device,
            hidden_dims=[256, 256],
            lr=LR,
            n_epochs=4,
            mini_batch_size=256,
            mix_type=cfg.get("mix_type"),
            mixer_dim=cfg.get("mixer_dim", 64),
            offline_critic_buffer=cfg.get("offline_critic", False),
        )
    if algo == "SAC":
        return SAC(
            obs_dim=OBS_DIM,
            continuous_action_dim=CONT_DIM,
            discrete_action_dims=DISC_DIMS,
            max_actions=MAX_ACTIONS,
            min_actions=MIN_ACTIONS,
            device=device,
            hidden_dims=[256, 256],
            lr=LR,
            mode=cfg.get("mode"),
        )
    if algo == "DQN":
        return DQN(
            obs_dim=OBS_DIM,
            continuous_action_dims=CONT_DIM,  # note: DQN uses plural
            discrete_action_dims=DISC_DIMS,
            max_actions=MAX_ACTIONS,
            min_actions=MIN_ACTIONS,
            device=device,
            hidden_dims=[256, 256],
            lr=LR,
            n_c_action_bins=N_DISC,  # 5 bins → same levels as discrete
            mix_type=cfg.get("mix_type"),
        )
    raise ValueError(f"Unknown algo: {algo}")


def _to_numpy(x):
    if x is None:
        return None
    return x.cpu().numpy() if th.is_tensor(x) else np.asarray(x)


def _build_env_action(disc_actions, cont_actions):
    """Map model outputs → [NUM_ENVS, 2] action array for the env."""
    d = _to_numpy(disc_actions).reshape(NUM_ENVS).astype(int)
    c = _to_numpy(cont_actions).reshape(NUM_ENVS).astype(np.float32)
    return np.stack([ACTION_LEVELS[d], c], axis=-1)  # [NUM_ENVS, 2]


def _pack_actions(disc_actions, cont_actions):
    """Pack into buffer storage: [NUM_ENVS, 1, 2] with [disc_idx, cont_val]."""
    d = _to_numpy(disc_actions).reshape(NUM_ENVS, 1, 1).astype(np.float32)
    c = _to_numpy(cont_actions).reshape(NUM_ENVS, 1, 1).astype(np.float32)
    return np.concatenate([d, c], axis=-1)  # [NUM_ENVS, 1, 2]


def run_lander_experiment(cfg_name, cfg, seed):
    algo = cfg["algo"]
    is_ppo = algo == "PPO"
    total_steps = STEPS_ON_POLICY if is_ppo else STEPS_OFF_POLICY
    device = "cuda" if th.cuda.is_available() else "cpu"
    full_gpu = device == "cuda"

    print(f"Running [{cfg_name}_seed{seed}] on {device} for {total_steps:,} steps...")

    log_dir = f"./lander_results/runs/lander_{cfg_name}_seed{seed}"
    writer = SummaryWriter(log_dir)

    env = envpool.make_gymnasium(
        "LunarLanderContinuous-v2", num_envs=NUM_ENVS, seed=seed
    )
    agent = _make_model(cfg, device)

    if is_ppo:
        # PPO log_prob layout (matches PG.reinforcement_learn reader):
        #   col 0 = continuous log prob, col 1+ = discrete log probs
        buffer = RolloutBuffer(
            buffer_size=2048 // NUM_ENVS,
            obs_shape=(OBS_DIM,),
            action_dim=ACTION_DIM,
            device=device,
            n_envs=NUM_ENVS,
            n_agents=1,
            log_probs_dim=ACTION_DIM,  # 1 cont + 1 disc = 2
            full_gpu=full_gpu,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size=100_000,
            obs_shape=(OBS_DIM,),
            action_dim=ACTION_DIM,
            device=device,
            n_envs=NUM_ENVS,
            n_agents=1,
            full_gpu=full_gpu,
        )

    obs, _ = env.reset()
    obs_t = th.as_tensor(obs, device=device).unsqueeze(1).float()  # [E,1,8]

    steps_done = 0
    ep_num = 0
    episode_rewards = np.zeros(NUM_ENVS)
    reward_curve = []
    warmup = 5_000 if not is_ppo else 0

    while steps_done < total_steps:
        agent.eval()
        with th.no_grad():
            act_dict = agent.train_actions(obs_t)

        disc_actions = act_dict["discrete_actions"]  # [E, 1, 1]
        cont_actions = act_dict["continuous_actions"]  # [E, 1, 1]

        env_actions = _build_env_action(disc_actions, cont_actions)  # [E, 2]
        actions_packed = _pack_actions(disc_actions, cont_actions)  # [E, 1, 2]

        obs_next, rewards, terminations, truncations, _ = env.step(env_actions)
        obs_next_t = th.as_tensor(obs_next, device=device).unsqueeze(1).float()

        if is_ppo:
            # Pack log_probs: continuous first, then discrete (PG reader order)
            c_lp = _to_numpy(act_dict["continuous_log_probs"]).reshape(NUM_ENVS, 1, 1)
            d_lp = _to_numpy(act_dict["discrete_log_probs"]).reshape(NUM_ENVS, 1, 1)
            log_probs = np.concatenate([c_lp, d_lp], axis=-1).astype(np.float32)
            buffer.add(
                obs=obs_t,
                action=th.as_tensor(actions_packed, device=device),
                reward=rewards[np.newaxis, ...],
                termination=terminations[np.newaxis, ...],
                truncation=truncations[np.newaxis, ...],
                value=act_dict["values"],
                log_prob=th.as_tensor(log_probs, device=device),
            )
        else:
            buffer.add(
                obs=obs_t,
                next_obs=obs_next_t,
                action=th.as_tensor(actions_packed, device=device),
                reward=rewards[np.newaxis, ...],
                term=terminations[np.newaxis, ...],
                trunc=truncations[np.newaxis, ...],
            )

        episode_rewards += rewards
        for i in range(NUM_ENVS):
            if terminations[i] or truncations[i]:
                reward_curve.append((steps_done, float(episode_rewards[i])))
                writer.add_scalar("Reward/episode", float(episode_rewards[i]), ep_num)
                ep_num += 1
                episode_rewards[i] = 0.0

        obs = obs_next
        obs_t = obs_next_t
        steps_done += NUM_ENVS

        if is_ppo and buffer.full:
            agent.train()
            with th.no_grad():
                last_values = agent.expected_V(obs_t)
            agent.reinforcement_learn(
                buffer,
                last_values,
                terminations[np.newaxis, ...],
                truncations[np.newaxis, ...],
            )
            buffer.reset()

        elif not is_ppo and steps_done > warmup:
            agent.train()
            for _ in range(UPDATES_PER_STEP):
                agent.reinforcement_learn(buffer.sample(BATCH_SIZE))

    env.close()
    writer.close()
    return {"reward_curve": reward_curve}


def _ema(y: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def main():
    os.makedirs("./lander_results", exist_ok=True)
    all_results = {}

    for cfg_name, cfg in CONFIGS.items():
        all_results[cfg_name] = []
        for seed in SEEDS:
            res = run_lander_experiment(cfg_name, cfg, seed)
            all_results[cfg_name].append(res)

    with open("./lander_results/lander_results.json", "w") as f:
        json.dump(all_results, f)

    # Plot — each algo family gets its own x-axis limit
    plt.figure(figsize=(12, 7))
    for cfg_name, runs in all_results.items():
        total_steps = STEPS_ON_POLICY if "PPO" in cfg_name else STEPS_OFF_POLICY
        all_curves = []
        for r in runs:
            curve = np.array(r["reward_curve"])
            if len(curve) > 0:
                x = np.linspace(0, total_steps, 200)
                y = _ema(np.interp(x, curve[:, 0], curve[:, 1]))
                all_curves.append(y)
        if all_curves:
            matrix = np.stack(all_curves)
            x_plot = np.linspace(0, total_steps, 200)
            mean, std = matrix.mean(0), matrix.std(0)
            plt.plot(x_plot, mean, label=cfg_name)
            plt.fill_between(x_plot, mean - std, mean + std, alpha=0.2)

    plt.legend(fontsize=8)
    plt.grid()
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward")
    plt.title("LunarLander Hybrid Action Benchmark (3 seeds)")
    plt.tight_layout()
    plt.savefig("./lander_results/lander_rewards.png", dpi=150)
    print("Saved lander_results/lander_rewards.png")


if __name__ == "__main__":
    main()
