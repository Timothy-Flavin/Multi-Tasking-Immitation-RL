import gymnasium as gym
import numpy as np
import time
import torch
import os
import sys

# Add src to path to import flexibuddiesrl
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from flexibuddiesrl.PG_stabalized import PG
from flexibuddiesrl.SAC import SAC
from flexibuddiesrl.DQN import DQN
from flexibuff import FlexibleBuffer
import matplotlib.pyplot as plt


class ContinuousCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def step(self, action):
        if isinstance(action, (list, np.ndarray)):
            action_val = action[0]
        else:
            action_val = action
        discrete_action = 1 if action_val > 0 else 0
        return self.env.step(discrete_action)


class BatchWrapper:
    def __init__(self, batch, clp_name):
        self.batch = batch
        self.clp_name = clp_name

    def __getattr__(self, name):
        val = getattr(self.batch, name)
        if name == self.clp_name and val is not None and isinstance(val, torch.Tensor):
            # Shape is (n_agents, batch, dim) if sampled from FlexibleBuffer
            # PG_stabalized expects (batch, dim) after agent indexing,
            # and then (batch,) for log probs of single action.
            # But wait, PG_stabalized does batch.clp[agent_num] which is (batch, dim)
            return val.squeeze(-1)
        return val


def run_profiling(agent_type, mode, total_steps=50000, device=None):
    print(f"\n--- Profiling {agent_type} ({mode}) ---")

    # Setup environment
    env = gym.make("CartPole-v1")
    is_continuous = mode == "continuous"
    if is_continuous:
        env = ContinuousCartPole(env)

    obs_dim = env.observation_space.shape[0]

    # Setup agent
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    common_kwargs = dict(
        obs_dim=obs_dim,
        device=device,
        hidden_dims=[64, 64],
    )

    if is_continuous:
        discrete_action_dims = None
        continuous_action_dim = 1
        max_actions = np.array([1.0], dtype=np.float32)
        min_actions = np.array([-1.0], dtype=np.float32)
    else:
        discrete_action_dims = [2]
        continuous_action_dim = 0
        max_actions = np.array([0.0], dtype=np.float32)
        min_actions = np.array([0.0], dtype=np.float32)

    if agent_type == "PPO":
        agent = PG(
            **common_kwargs,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            ppo_clip=0.2,
            entropy_loss=0.01,
            value_clip=0.5,
            advantage_type="gae",
            n_epochs=4,
            mini_batch_size=128,
        )
        batch_size = 2048
        update_every = 2048
    elif agent_type == "SAC":
        agent = SAC(
            **common_kwargs,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
        )
        batch_size = 256
        update_every = 8
    elif agent_type == "DQN":
        agent = DQN(
            **common_kwargs,
            continuous_action_dims=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            n_c_action_bins=5,
        )
        batch_size = 256
        update_every = 8

    # Setup buffer
    n_discrete_heads = len(discrete_action_dims) if discrete_action_dims else 0
    buffer = FlexibleBuffer(
        num_steps=10000,
        n_agents=1,
        discrete_action_cardinalities=discrete_action_dims,
        track_action_mask=False,
        path=f"./profiling_buffer_{agent_type}_{mode}",
        name="prof_buffer",
        memory_weights=False,
        global_registered_vars={
            "global_rewards": (None, np.float32),
        },
        individual_registered_vars={
            "obs": ([obs_dim], np.float32),
            "obs_": ([obs_dim], np.float32),
            "discrete_log_probs": ([n_discrete_heads], np.float32),
            "continuous_log_probs": ([continuous_action_dim], np.float32),
            "discrete_actions": ([n_discrete_heads], np.int64),
            "continuous_actions": ([continuous_action_dim], np.float32),
        },
    )

    action_times = []
    update_times = []
    rewards = []
    current_episode_reward = 0.0

    obs, _ = env.reset()

    for step in range(total_steps):
        # Taking Action
        start_time = time.perf_counter()
        obs_f = obs.astype(np.float32)
        act_dict = agent.train_actions(obs_f, step=True)
        action_times.append(time.perf_counter() - start_time)

        dact = act_dict.get("discrete_actions")
        cact = act_dict.get("continuous_actions")
        dlp = act_dict.get("discrete_log_probs")
        clp = act_dict.get("continuous_log_probs")

        if dact is None:
            dact = np.array([], dtype=np.int64)
        if cact is None:
            cact = np.array([], dtype=np.float32)
        if dlp is None:
            dlp = np.array([], dtype=np.float32)
        if clp is None:
            clp = np.array([], dtype=np.float32)

        if is_continuous:
            env_action = cact
        else:
            env_action = int(dact[0])

        obs_, reward, terminated, truncated, _ = env.step(env_action)
        current_episode_reward += reward

        # Buffer save
        rv = {
            "global_rewards": float(reward),
            "obs": [obs_f.copy()],
            "obs_": [obs_.astype(np.float32).copy()],
            "discrete_log_probs": [
                (
                    dlp.flatten()
                    if dlp.size > 0
                    else np.zeros(n_discrete_heads, dtype=np.float32)
                )
            ],
            "continuous_log_probs": [
                (
                    clp.flatten()
                    if clp.size > 0
                    else np.zeros(continuous_action_dim, dtype=np.float32)
                )
            ],
            "discrete_actions": [
                (
                    dact.flatten()
                    if dact.size > 0
                    else np.zeros(n_discrete_heads, dtype=np.int64)
                )
            ],
            "continuous_actions": [
                (
                    cact.flatten()
                    if cact.size > 0
                    else np.zeros(continuous_action_dim, dtype=np.float32)
                )
            ],
        }
        buffer.save_transition(terminated=terminated, registered_vals=rv)

        obs = obs_
        if terminated or truncated:
            obs, _ = env.reset()
            rewards.append(current_episode_reward)
            current_episode_reward = 0.0

        if (step + 1) % 10000 == 0:
            print(f"  Step {step+1}/{total_steps}...")

        # Updating
        if (step + 1) % update_every == 0 and (step + 1) >= batch_size:
            start_time = time.perf_counter()
            if agent_type == "PPO":
                mb = buffer.sample_transitions(
                    idx=np.arange(0, batch_size), as_torch=True, device=device
                )
                if is_continuous:
                    clp_name = agent.batch_name_map["continuous_log_probs"]
                    mb = BatchWrapper(mb, clp_name)
                agent.reinforcement_learn(mb, 0)
                buffer.reset()
            else:
                mb = buffer.sample_transitions(batch_size, as_torch=True, device=device)
                agent.reinforcement_learn(mb, 0)
            update_times.append(time.perf_counter() - start_time)

    total_action_time = sum(action_times)
    total_update_time = sum(update_times)
    avg_action_time = total_action_time / total_steps
    avg_update_time = total_update_time / len(update_times) if update_times else 0

    print(f"Total steps: {total_steps}")
    print(
        f"Total time taking actions: {total_action_time:.4f}s (Avg: {avg_action_time*1000:.4f}ms)"
    )
    print(
        f"Total time updating: {total_update_time:.4f}s (Avg: {avg_update_time*1000:.4f}ms)"
    )
    print(f"Updates performed: {len(update_times)}")

    # Save plot of rewards over time
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{agent_type} ({mode}) Rewards Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"{agent_type}_{mode}_{device}_rewards.png")
    plt.close()

    return {
        "agent": agent_type,
        "mode": mode,
        "total_action_time": total_action_time,
        "total_update_time": total_update_time,
        "avg_action_time": avg_action_time,
        "avg_update_time": avg_update_time,
        "updates": len(update_times),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu, cuda, etc.)"
    )
    args = parser.parse_args()

    results = []

    experiments = [
        ("PPO", "discrete"),
        ("PPO", "continuous"),
        ("SAC", "discrete"),
        ("SAC", "continuous"),
        ("DQN", "discrete"),
        ("DQN", "continuous"),
    ]

    for agent, mode in experiments:
        try:
            sp = time.time()
            res = run_profiling(agent, mode, total_steps=args.steps, device=args.device)
            print(f"Experiment {agent} ({mode}) completed in {time.time() - sp:.2f}s")
            results.append(res)
        except Exception as e:
            print(f"Failed to profile {agent} ({mode}): {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(
        f"{'Agent':<10} {'Mode':<12} {'Action Total':<15} {'Action Avg(ms)':<15} {'Update Total':<15} {'Update Avg(ms)':<15}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['agent']:<10} {r['mode']:<12} {r['total_action_time']:<15.4f} {r['avg_action_time']*1000:<15.4f} {r['total_update_time']:<15.4f} {r['avg_update_time']*1000:<15.4f}"
        )
    print("=" * 80)
