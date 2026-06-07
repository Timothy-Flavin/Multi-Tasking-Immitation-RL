"""
Integration test: 12+ configurations x 3 seeds on vectorized ContextualDecouplerEnv.
Uses the new tensor-based buffer implementations.
Logs to tensorboard under ./dependence_results/runs/vectorized_<config>_seed<s>/
"""

import gymnasium as gym
import numpy as np
import torch as th
import torch
import os
import json
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from flexibuddiesrl.PG_new_buffer import PG
from flexibuddiesrl.SAC_new_buffer import SAC
from flexibuddiesrl.DQN_new_buffer import DQN
from flexibuddiesrl.buffers import RolloutBuffer, ReplayBuffer
from toy_env import ContextualDecouplerEnv

# -- Hyper-parameters ---------------------------------------------------------
N_ACTIONS = 5
OBS_DIM = 3
TOTAL_STEPS = 100_000
NUM_ENVS = 16
BATCH_SIZE = 256
LR = 1e-3
SEEDS = [0, 1, 2]

CONFIGS = {
    "PPO_shared_nomix": dict(algo="PPO", mix_type=None),
    "PPO_VDN": dict(algo="PPO", mix_type="VDN"),
    "PPO_QMIX": dict(algo="PPO", mix_type="QMIX", mixer_dim=64),
    "SAC_Q": dict(algo="SAC", mode="Q"),
    "SAC_V": dict(algo="SAC", mode="V"),
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
        kw = common.copy()
        kw.update(dict(mode=cfg.get("mode"), target_discrete_entropy_percentage=0.1))
        return SAC(**kw)
    elif algo == "DQN":
        kw = common.copy()
        kw["continuous_action_dims"] = kw.pop("continuous_action_dim")
        kw.update(dict(mix_type=cfg.get("mix_type"), n_c_action_bins=5))
        return DQN(**kw)
    raise ValueError(f"Unknown algo: {algo}")


def run_vectorized_experiment(cfg_name, cfg, seed):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Running [{cfg_name}_seed{seed}] on {device}...")

    log_dir = f"./dependence_results/runs/vectorized_{cfg_name}_seed{seed}"
    writer = SummaryWriter(log_dir)

    # Use a vectorized environment
    env = gym.vector.SyncVectorEnv(
        [lambda: ContextualDecouplerEnv() for _ in range(NUM_ENVS)]
    )

    agent = _make_model(cfg, device)
    is_ppo = cfg["algo"] == "PPO"

    # Setup Buffer
    if is_ppo:
        buffer = RolloutBuffer(
            buffer_size=2048 // NUM_ENVS,
            obs_shape=(OBS_DIM,),
            action_dim=2,
            device=device,
            n_envs=NUM_ENVS,
            n_agents=1,
            log_probs_dim=2,  # 2 discrete heads
            full_gpu=True,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size=50_000,
            obs_shape=(OBS_DIM,),
            action_dim=2,
            device=device,
            n_envs=NUM_ENVS,
            n_agents=1,
            full_gpu=True,
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

        d_actions_np = (
            d_actions.cpu().numpy() if torch.is_tensor(d_actions) else d_actions
        )
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
                # but results.json will store raw per-step importance for detail.
                importance_history.append((steps_done, rl["importance_raw"].tolist()))

        elif not is_ppo and steps_done > 1000 and steps_done % 1 == 0:
            agent.train()
            samples = buffer.sample(BATCH_SIZE)
            rl = agent.reinforcement_learn(samples)
            updates += 1
            if "importance_raw" in rl:
                # ReplayBuffer samples are shuffled, so "steps_done" is just an update index here.
                importance_history.append((steps_done, rl["importance_raw"].tolist()))

    env.close()
    writer.close()
    return {"reward_curve": reward_curve, "importance_history": importance_history}


def debug_critic_learning(n_steps=50_000, batch_size=256, seed=0, probe_interval=5_000):
    """
    Fills a replay buffer with random actions and trains only the SAC V-mode critic.
    Periodically probes Q(s, a) at hand-picked states to check whether the critic
    is learning the right value function before the actor is involved at all.

    The key diagnostic is the Q-gap between optimal and wrong actions, which should
    converge to ~2.0 (the reward difference from a single correct vs. incorrect step).
    If the gap stays near 0, the critic has a bug. If the gap is ~2.0 but the agent
    still fails to learn, the problem is in the actor loss.
    """
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"\n=== Critic-Only Debug | SAC V mode | device={device} ===")

    rng = np.random.default_rng(seed)
    env = ContextualDecouplerEnv()

    agent = SAC(
        obs_dim=OBS_DIM,
        continuous_action_dim=0,
        discrete_action_dims=[N_ACTIONS, N_ACTIONS],
        device=device,
        hidden_dims=[64, 64],
        lr=LR,
        mode="V",
    )

    buffer = ReplayBuffer(
        buffer_size=50_000,
        obs_shape=(OBS_DIM,),
        action_dim=2,
        device=device,
        n_envs=1,
        n_agents=1,
        full_gpu=(device == "cuda"),
    )

    # Hand-picked probe states: [context, target_0, target_1]
    # Each pair shows optimal vs. wrong action for that context.
    # Q_gap = Q(optimal) - Q(wrong) should converge to ~2.0 (one-step reward delta).
    probes = [
        (np.array([0, 2, 3]), np.array([2, 0]), "ctx=0 t0=2 | a=(2,0) OPTIMAL"),
        (np.array([0, 2, 3]), np.array([3, 0]), "ctx=0 t0=2 | a=(3,0) WRONG  "),
        (np.array([1, 2, 3]), np.array([0, 3]), "ctx=1 t1=3 | a=(0,3) OPTIMAL"),
        (np.array([1, 2, 3]), np.array([0, 2]), "ctx=1 t1=3 | a=(0,2) WRONG  "),
    ]
    probe_obs = th.tensor(
        np.stack([p[0] for p in probes]), dtype=th.float32, device=device
    )
    probe_acts = th.tensor(
        np.stack([p[1] for p in probes]), dtype=th.long, device=device
    )
    probe_labels = [p[2] for p in probes]

    obs, _ = env.reset(seed=seed)
    update_count = 0
    critic_loss = float("nan")

    for step in range(1, n_steps + 1):
        action = env.action_space.sample()
        obs_next, reward, terminated, truncated, _ = env.step(action)

        buffer.add(
            obs=obs.astype(np.float32).reshape(1, 1, OBS_DIM),
            next_obs=obs_next.astype(np.float32).reshape(1, 1, OBS_DIM),
            action=np.array(action, dtype=np.float32).reshape(1, 1, 2),
            reward=np.array([[reward]], dtype=np.float32),
            term=np.array([[terminated]], dtype=np.float32),
            trunc=np.array([[truncated]], dtype=np.float32),
        )

        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        else:
            obs = obs_next

        if step > 1000:
            agent.train()
            samples = buffer.sample(batch_size)
            rl = agent.reinforcement_learn(samples, critic_only=True)
            critic_loss = rl["critic_loss"]
            update_count += 1

        if step % probe_interval == 0:
            agent.eval()
            with th.no_grad():
                q_vals = agent.utility_function(probe_obs, actions=(probe_acts, None))
            q_vals = q_vals.cpu().tolist()
            print(
                f"\n[step={step:>6d}  updates={update_count:>5d}  critic_loss={critic_loss:.4f}]"
            )
            for label, q in zip(probe_labels, q_vals):
                print(f"  Q {label} = {q:+.4f}")
            gap_ctx0 = q_vals[0] - q_vals[1]
            gap_ctx1 = q_vals[2] - q_vals[3]
            print(f"  --> Q_gap ctx=0 (expect ~2.0): {gap_ctx0:+.4f}")
            print(f"  --> Q_gap ctx=1 (expect ~2.0): {gap_ctx1:+.4f}")

    env.close()
    print("\n=== Done ===")


def main():
    os.makedirs("./dependence_results", exist_ok=True)
    all_results = {}

    for cfg_name, cfg in CONFIGS.items():
        all_results[cfg_name] = []
        for seed in SEEDS:
            res = run_vectorized_experiment(cfg_name, cfg, seed)
            all_results[cfg_name].append(res)

    with open("./dependence_results/vectorized_results.json", "w") as f:
        json.dump(all_results, f)

    # Plot results
    plt.figure(figsize=(10, 6))
    for cfg_name, runs in all_results.items():
        all_curves = []
        for r in runs:
            curve = np.array(r["reward_curve"])
            if len(curve) > 0:
                # Bin to common x-axis
                x = np.linspace(0, TOTAL_STEPS, 100)
                y = np.interp(x, curve[:, 0], curve[:, 1])
                all_curves.append(y)

        if all_curves:
            matrix = np.stack(all_curves)
            plt.plot(x, matrix.mean(0), label=cfg_name)
            plt.fill_between(
                x,
                matrix.mean(0) - matrix.std(0),
                matrix.mean(0) + matrix.std(0),
                alpha=0.2,
            )
    plt.legend()
    plt.grid()
    plt.title(f"Vectorized Reward Curves (100k steps)")
    plt.savefig("vectorized_rewards.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug_critic":
        debug_critic_learning()
    else:
        main()
