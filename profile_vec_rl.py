import gymnasium as gym
import numpy as np
import time
import torch as th
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch

# Try to import envpool, providing instructions if it fails
try:
    import envpool
except ImportError:
    print("\nError: 'envpool' is not installed.")
    print("Please install it using: pip install envpool\n")
    sys.exit(1)

# Add src to path to import flexibuddiesrl
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from flexibuddiesrl.PG_new_buffer import PG
from flexibuddiesrl.SAC_new_buffer import SAC
from flexibuddiesrl.DQN_new_buffer import DQN
from flexibuddiesrl.buffers import RolloutBuffer, ReplayBuffer


class VectorizedContinuousCartPole:
    """Wrapper to handle continuous actions for envpool CartPole."""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def step(self, action):
        # action is [num_envs, 1]
        discrete_action = (action > 0).astype(np.int32).flatten()
        return self.env.step(discrete_action)

    def reset(self):
        return self.env.reset()


def run_profiling_new(
    agent_type, mode, total_steps=200000, num_envs=64, device="auto", full_gpu=False
):
    print(
        f"\n--- Profiling {agent_type} ({mode}) with EnvPool ({num_envs} envs) | full_gpu={full_gpu} ---"
    )

    device = th.device(
        "cuda"
        if th.cuda.is_available() and device == "auto"
        else ("cpu" if device == "auto" else device)
    )
    print(f"Using device: {device}")

    # Setup environment
    env_name = "CartPole-v1"
    raw_env = envpool.make(env_name, env_type="gymnasium", num_envs=num_envs)
    is_continuous = mode == "continuous"

    obs_dim = raw_env.observation_space.shape[0]

    # Common action parameters
    discrete_action_dims = [2]
    continuous_action_dim = 1
    max_actions = np.array([1.0], dtype=np.float32)
    min_actions = np.array([-1.0], dtype=np.float32)

    # Standardize update ratio: 1 update per 8 environment steps
    # For 64 envs, this means 64/8 = 8 updates per parallel step
    update_every = 8
    num_updates_per_step = num_envs // update_every
    batch_size = 256

    if agent_type == "PPO":
        agent = PG(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dim if is_continuous else 0,
            discrete_action_dims=None if is_continuous else discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            device=device,
            hidden_dims=[64, 64],
            n_epochs=4,  # Rollout buffer size transitions, 4 epochs
            mini_batch_size=128,
            lr=3e-4,
        )
        # 2048 transitions / 64 envs = 32 parallel steps
        collection_steps = 2048 // num_envs
        if collection_steps < 1:
            collection_steps = 1

        lp_dim = 1 if is_continuous else len(discrete_action_dims)
        buffer = RolloutBuffer(
            buffer_size=collection_steps,
            n_envs=num_envs,
            n_agents=1,
            obs_shape=(obs_dim,),
            action_dim=(
                continuous_action_dim if is_continuous else len(discrete_action_dims)
            ),
            device=device,
            log_probs_dim=lp_dim,
            full_gpu=full_gpu,
            action_dtype=torch.float32 if is_continuous else torch.int32,
        )
    elif agent_type == "SAC":
        agent = SAC(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dim if is_continuous else 0,
            discrete_action_dims=None if is_continuous else discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            device=device,
            hidden_dims=[64, 64],
            lr=1e-3,
        )
        buffer = ReplayBuffer(
            buffer_size=1000000,
            n_envs=num_envs,
            n_agents=1,
            obs_shape=(obs_dim,),
            action_dim=(
                continuous_action_dim if is_continuous else len(discrete_action_dims)
            ),
            device=device,
            full_gpu=full_gpu,
        )
    elif agent_type == "DQN":
        agent = DQN(
            obs_dim=obs_dim,
            continuous_action_dims=continuous_action_dim if is_continuous else 0,
            discrete_action_dims=None if is_continuous else discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            device=device,
            hidden_dims=[64, 64],
            lr=1e-3,
        )
        buffer = ReplayBuffer(
            buffer_size=1000000,
            n_envs=num_envs,
            n_agents=1,
            obs_shape=(obs_dim,),
            action_dim=(
                continuous_action_dim if is_continuous else len(discrete_action_dims)
            ),
            device=device,
            full_gpu=full_gpu,
        )

    start_total = time.perf_counter()
    steps_done = 0
    last_print_step = 0
    obs = raw_env.reset()[0]

    episode_rewards = np.zeros(num_envs)
    rewards_history = []

    while steps_done < total_steps:
        # 1. Experience Collection
        agent.eval()
        if agent_type == "PPO":
            for _ in range(collection_steps):
                with th.no_grad():
                    obs_t = th.as_tensor(obs, device=device).unsqueeze(0)
                    act_dict = agent.train_actions(obs_t)
                    values = th.as_tensor(act_dict["values"], device=device)
                    log_probs = th.as_tensor(
                        act_dict[
                            (
                                "continuous_log_probs"
                                if is_continuous
                                else "discrete_log_probs"
                            )
                        ],
                        device=device,
                    )
                    raw_actions = act_dict[
                        ("continuous_actions" if is_continuous else "discrete_actions")
                    ]

                # Env Step
                if is_continuous:
                    env_actions = raw_actions.reshape(num_envs, 1).astype(np.float32)
                    # Mapping continuous to discrete for CartPole
                    obs_next, rewards, terminations, truncations, info = raw_env.step(
                        (env_actions > 0).astype(np.int32).flatten()
                    )
                else:
                    env_actions = raw_actions.flatten()
                    obs_next, rewards, terminations, truncations, info = raw_env.step(
                        env_actions
                    )

                # Handling Truncation
                final_obs = None
                if np.any(truncations):
                    final_obs = np.zeros_like(obs)
                    f_obs_src = (
                        info["final_observation"]
                        if "final_observation" in info
                        else (info["obs"] if "obs" in info else None)
                    )
                    if f_obs_src is not None:
                        for i in range(num_envs):
                            if truncations[i]:
                                final_obs[i] = f_obs_src[i]
                    else:
                        final_obs = obs_next
                    final_obs = final_obs[np.newaxis, ...]

                buffer.add(
                    obs=obs[np.newaxis, ...],
                    action=raw_actions,
                    reward=rewards[np.newaxis, ...],
                    termination=terminations[np.newaxis, ...],
                    truncation=truncations[np.newaxis, ...],
                    value=values,
                    log_prob=log_probs,
                    final_obs=final_obs,
                )

                episode_rewards += rewards
                for i in range(num_envs):
                    if terminations[i] or truncations[i]:
                        rewards_history.append(episode_rewards[i])
                        episode_rewards[i] = 0
                obs = obs_next
                steps_done += num_envs

            # PPO Update
            agent.train()
            with th.no_grad():
                last_obs_t = th.as_tensor(obs, device=device).unsqueeze(0)
                last_values = agent.expected_V(last_obs_t)
            agent.reinforcement_learn(
                buffer,
                last_values,
                terminations[np.newaxis, ...],
                truncations[np.newaxis, ...],
            )
            buffer.reset()

        else:
            # Off-policy (SAC/DQN)
            with th.no_grad():
                obs_t = th.as_tensor(obs, device=device).unsqueeze(0)
                act_dict = agent.train_actions(obs_t)
                raw_actions = act_dict[
                    ("continuous_actions" if is_continuous else "discrete_actions")
                ]

            if is_continuous:
                env_actions = raw_actions.reshape(num_envs, 1).astype(np.float32)
                obs_next, rewards, terminations, truncations, info = raw_env.step(
                    (env_actions > 0).astype(np.int32).flatten()
                )
            else:
                env_actions = raw_actions.flatten()
                obs_next, rewards, terminations, truncations, info = raw_env.step(
                    env_actions
                )

            buffer.add(
                obs=obs[np.newaxis, ...],
                next_obs=obs_next[np.newaxis, ...],
                action=raw_actions,
                reward=rewards[np.newaxis, ...],
                term=terminations[np.newaxis, ...],
                trunc=truncations[np.newaxis, ...],
            )

            episode_rewards += rewards
            for i in range(num_envs):
                if terminations[i] or truncations[i]:
                    rewards_history.append(episode_rewards[i])
                    episode_rewards[i] = 0
            obs = obs_next
            steps_done += num_envs

            # Scaling Off-Policy Updates
            if steps_done >= 1000:
                agent.train()
                for _ in range(num_updates_per_step):
                    samples = buffer.sample(batch_size)
                    agent.reinforcement_learn(samples, agent_num=0)

        if (steps_done - last_print_step) >= 20000:
            last_print_step = steps_done
            avg_rew = np.mean(rewards_history[-100:]) if len(rewards_history) > 0 else 0
            print(f"Step: {steps_done:<8} | Avg Reward: {avg_rew:.2f}")

    end_total = time.perf_counter()
    duration = end_total - start_total
    sps = steps_done / duration

    print(f"Profiling Results: {sps:.2f} SPS | Duration: {duration:.4f}s")
    os.makedirs("profile_results", exist_ok=True)
    np.save(
        f"profile_results/rewards_{agent_type}_{mode}_{device.type}_fullgpu_{full_gpu}.npy",
        np.array(rewards_history),
    )
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{agent_type} ({mode}) EnvPool Rewards | GPU={full_gpu}")
    plt.grid()
    plt.savefig(
        f"profile_results/new_profiler_{agent_type}_{mode}_{device.type}_fullgpu_{full_gpu}.png"
    )
    plt.close()

    return {
        "agent": agent_type,
        "mode": mode,
        "sps": sps,
        "duration": duration,
        "full_gpu": full_gpu,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    final_results = []

    algos = ["PPO", "SAC", "DQN"]
    modes = ["discrete", "continuous"]
    gpu_options = [False, True]

    for full_gpu in gpu_options:
        for agent_type in algos:
            for mode in modes:
                try:
                    sp = time.time()
                    res = run_profiling_new(
                        agent_type,
                        mode,
                        total_steps=args.steps,
                        num_envs=args.envs,
                        device=args.device,
                        full_gpu=full_gpu,
                    )
                    print(
                        f"Experiment {agent_type} ({mode}) with full_gpu={full_gpu} completed in {time.time() - sp:.2f}s"
                    )
                    final_results.append(res)
                except Exception as e:
                    print(
                        f"Failed to profile {agent_type} ({mode}) full_gpu={full_gpu}: {e}"
                    )
                    import traceback

                    traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"{'Agent':<10} {'Mode':<12} {'FullGPU':<10} {'SPS':<15} {'Duration(s)':<15}")
    print("-" * 80)
    for r in final_results:
        print(
            f"{r['agent']:<10} {r['mode']:<12} {str(r['full_gpu']):<10} {r['sps']:<15.2f} {r['duration']:<15.4f}"
        )
    print("=" * 80)
