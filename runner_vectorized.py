"""
Integration test: 12+ configurations x 3 seeds on vectorized ContextualDecouplerEnv.
Uses the new tensor-based buffer implementations.
Logs to tensorboard under ./dependence_results/runs/vectorized_<config>_seed<s>/
"""

import sys, os
import gymnasium as gym
import numpy as np
import torch
import torch as th
import time
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from flexibuddiesrl.PG_new_buffer import PG
from flexibuddiesrl.SAC_new_buffer import SAC
from flexibuddiesrl.DQN_new_buffer import DQN
from flexibuddiesrl.buffers import RolloutBuffer, ReplayBuffer
from toy_env import ContextualDecouplerEnv

# -- Hyper-parameters ---------------------------------------------------------
N_ACTIONS = 5
OBS_DIM = 3
TOTAL_STEPS = 50_000
NUM_ENVS = 16
BATCH_SIZE = 256
LR = 1e-3
SEEDS = [0, 1, 2]

CONFIGS = {
    "PPO_shared_nomix": dict(algo="PPO", mix_type=None),
    "PPO_VDN": dict(algo="PPO", mix_type="VDN"),
    "PPO_QMIX": dict(algo="PPO", mix_type="QMIX", mixer_dim=64),
    "SAC_shared_nomix": dict(algo="SAC"),
    "DQN_shared_nomix": dict(algo="DQN", mix_type=None),
    "DQN_VDN": dict(algo="DQN", mix_type="VDN"),
    "DQN_QMIX": dict(algo="DQN", mix_type="QMIX", mixer_dim=64),
}


def _make_model(cfg, device):
    algo = cfg["algo"]
    common = dict(
        obs_dim=OBS_DIM,
        continuous_action_dim=0,
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        device=device,
        hidden_dims=[64, 64],
        lr=LR,
    )
    if algo == "PPO":
        kw = common.copy()
        kw.update(dict(n_epochs=4, mini_batch_size=128, mix_type=cfg.get("mix_type")))
        return PG(**kw)
    elif algo == "SAC":
        return SAC(**common)
    elif algo == "DQN":
        kw = common.copy()
        kw["continuous_action_dims"] = kw.pop("continuous_action_dim")
        kw.update(dict(mix_type=cfg.get("mix_type"), n_c_action_bins=5))
        return DQN(**kw)


def run_vectorized_experiment(config_name, cfg, seed):
    tag = f"vectorized_{config_name}_seed{seed}"
    log_dir = os.path.join("dependence_results", "runs", tag)
    writer = SummaryWriter(log_dir=log_dir)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    np.random.seed(seed)
    th.manual_seed(seed)

    # Create vectorized environment
    env = gym.vector.SyncVectorEnv(
        [lambda: ContextualDecouplerEnv(n_actions=N_ACTIONS) for _ in range(NUM_ENVS)]
    )

    agent = _make_model(cfg, device)
    is_ppo = cfg["algo"] == "PPO"

    # Setup Buffer
    if is_ppo:
        collection_steps = 1024 // NUM_ENVS
        buffer = RolloutBuffer(
            buffer_size=collection_steps,
            n_envs=NUM_ENVS,
            n_agents=1,
            obs_shape=(OBS_DIM,),
            action_dim=2,
            log_probs_dim=2,
            device=device,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size=100000,
            n_envs=NUM_ENVS,
            n_agents=1,
            obs_shape=(OBS_DIM,),
            action_dim=2,
            device=device,
        )

    obs, _ = env.reset(seed=seed)
    obs_t = th.as_tensor(obs, device=device).unsqueeze(1).float()

    steps_done, ep_num, updates = 0, 0, 0
    episode_rewards = np.zeros(NUM_ENVS)
    reward_curve, importance_history = [], []

    while steps_done < TOTAL_STEPS:
        agent.eval()
        with th.no_grad():
            act_dict = agent.train_actions(obs_t)
            d_actions = act_dict["discrete_actions"]  # [NUM_ENVS, 1, 2]

            if is_ppo:
                values = act_dict["values"]
                log_probs = act_dict["discrete_log_probs"]

        d_actions_np = d_actions.cpu().numpy() if torch.is_tensor(d_actions) else d_actions
        env_actions = d_actions_np.reshape(NUM_ENVS, 2).astype(int)
        obs_next, rewards, terminations, truncations, info = env.step(env_actions)

        # Convert ONLY what is needed for the next step or agent inference
        obs_next_t = th.as_tensor(obs_next, device=device).unsqueeze(1).float()

        # Capture context (first element of observation) for importance analysis
        # Using context from the *start* of the transition
        contexts = obs[:, 0].astype(int)

        if is_ppo:
            buffer.add(
                obs=obs_t if buffer.full_gpu else obs[np.newaxis, ...],
                action=d_actions if buffer.full_gpu else d_actions_np,
                reward=rewards[np.newaxis, ...],
                termination=terminations[np.newaxis, ...],
                truncation=truncations[np.newaxis, ...],
                value=values,
                log_prob=log_probs,
            )
        else:
            buffer.add(
                obs=obs_t if buffer.full_gpu else obs[np.newaxis, ...],
                next_obs=obs_next_t if buffer.full_gpu else obs_next[np.newaxis, ...],
                action=d_actions if buffer.full_gpu else d_actions_np,
                reward=rewards[np.newaxis, ...],
                term=terminations[np.newaxis, ...],
                trunc=truncations[np.newaxis, ...],
            )

        episode_rewards += rewards
        for i in range(NUM_ENVS):
            if terminations[i] or truncations[i]:
                reward_curve.append((steps_done, episode_rewards[i]))
                writer.add_scalar("Reward/episode", episode_rewards[i], ep_num)
                ep_num += 1
                episode_rewards[i] = 0

        obs = obs_next
        obs_t = obs_next_t
        steps_done += NUM_ENVS

        # Training
        if is_ppo and buffer.full:
            agent.train()
            with th.no_grad():
                last_values = agent.expected_V(obs_t)
            rl = agent.reinforcement_learn(
                buffer,
                last_values,
                terminations[np.newaxis, ...],
                truncations[np.newaxis, ...],
            )
            buffer.reset()
            updates += 1
            if "importance_raw" in rl:
                # Use mean context for importance summary in TB if needed,
                # but better to handle properly like baseline if possible.
                # For simplicity in vectorized:
                importance_history.append(
                    _analyze_importance(rl["importance_raw"], contexts, writer, updates)
                )
        elif not is_ppo and steps_done >= 1000:
            agent.train()
            # 1 update per 8 env steps ratio -> NUM_ENVS // 8 updates
            for _ in range(NUM_ENVS // 8):
                samples = buffer.sample(BATCH_SIZE)
                rl = agent.reinforcement_learn(samples)
                updates += 1
                if "importance_raw" in rl:
                    importance_history.append(
                        _analyze_importance(
                            rl["importance_raw"], contexts, writer, updates
                        )
                    )

    env.close()
    writer.close()
    print(f"  [{tag}] done")
    return {"reward_curve": reward_curve, "importance_history": importance_history}


def _analyze_importance(imp_raw, contexts, writer, step):
    # imp_raw is [B, n_heads]
    # In vectorized, we might not have a perfect 1-to-1 mapping of contexts to minibatch steps easily
    # without passing contexts to the buffer.
    # For now, let's assume we just want to see if correlation is positive.
    # PPO's importance_raw is already [n_steps, n_heads] for the whole buffer.
    return {"step": step}  # Simplified for now


def main():
    os.makedirs("dependence_results", exist_ok=True)
    results = {}
    for cfg_name, cfg in CONFIGS.items():
        results[cfg_name] = []
        for seed in SEEDS:
            res = run_vectorized_experiment(cfg_name, cfg, seed)
            results[cfg_name].append(res)

    with open("dependence_results/vectorized_results.json", "w") as f:
        json.dump(_serializable(results), f)
    # Reuse plot function from baseline if we can, or just plot here.
    _plot_reward_curves(results, "dependence_results/vectorized_")


def _serializable(results):
    res = {}
    for k, runs in results.items():
        res[k] = [
            {"reward_curve": [[float(s), float(v)] for s, v in r["reward_curve"]]}
            for r in runs
        ]
    return res


def _plot_reward_curves(results, prefix):
    plt.figure(figsize=(10, 6))
    for name, runs in results.items():
        curves = [r["reward_curve"] for r in runs if r["reward_curve"]]
        if not curves:
            continue
        steps = np.linspace(0, TOTAL_STEPS, 100)
        matrix = np.array(
            [
                np.interp(steps, [c[0] for c in curve], [c[1] for c in curve])
                for curve in curves
            ]
        )
        plt.plot(steps, matrix.mean(0), label=name)
        plt.fill_between(
            steps,
            matrix.mean(0) - matrix.std(0),
            matrix.mean(0) + matrix.std(0),
            alpha=0.2,
        )
    plt.legend()
    plt.grid()
    plt.title(f"{prefix} Reward Curves")
    plt.savefig(f"{prefix}rewards.png")


if __name__ == "__main__":
    main()
