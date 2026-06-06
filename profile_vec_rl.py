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


def _sync(device):
    """Flush GPU queue before reading the clock so timings capture completed work."""
    if device.type == "cuda":
        th.cuda.synchronize(device)


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
            n_epochs=4,
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
    elif agent_type == "SACQ":
        agent = SAC(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dim if is_continuous else 0,
            discrete_action_dims=None if is_continuous else discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            device=device,
            hidden_dims=[64, 64],
            lr=1e-3,
            mode="Q",
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
    elif agent_type == "SACV":
        agent = SAC(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dim if is_continuous else 0,
            discrete_action_dims=None if is_continuous else discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            device=device,
            hidden_dims=[64, 64],
            lr=1e-3,
            mode="V",
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

    # -------------------------------------------------------------------------
    # Per-phase timing accumulators
    # Each counter measures wall-clock seconds spent in that phase across the
    # entire run. GPU timing uses cuda.synchronize() before reading the clock so
    # the kernel queue is drained and the measurement reflects completed work.
    # -------------------------------------------------------------------------
    t_take_action = 0.0  # agent.train_actions (inference)
    t_env_step = 0.0  # raw_env.step (environment physics)
    t_buf_insert = 0.0  # buffer.add (host-side copy + optional H2D)
    t_net_update = 0.0  # agent.reinforcement_learn (gradient step)
    n_action_calls = 0
    n_update_calls = 0

    start_total = time.perf_counter()
    steps_done = 0
    last_print_step = 0
    obs = raw_env.reset()[0]
    obs_t = th.as_tensor(obs, device=device).unsqueeze(1)  # [E, A=1, obs_dim]

    episode_rewards = np.zeros(num_envs)
    rewards_history = []

    while steps_done < total_steps:
        if agent_type == "PPO":
            # --- PPO: collect a full rollout, then update ---
            agent.eval()
            for _ in range(collection_steps):
                # Phase 1: take_action
                _sync(device)
                _t = time.perf_counter()
                with th.no_grad():
                    # [E, A=1, obs_dim] — [E, A, O] layout per spec
                    act_dict = agent.train_actions(obs_t)
                    values = th.as_tensor(act_dict["values"], device=device)  # [E, A=1]
                    log_probs = th.as_tensor(
                        act_dict[
                            (
                                "continuous_log_probs"
                                if is_continuous
                                else "discrete_log_probs"
                            )
                        ],
                        device=device,
                    )  # [E, A=1, lp_dim]
                    raw_actions = act_dict[
                        "continuous_actions" if is_continuous else "discrete_actions"
                    ]
                _sync(device)
                t_take_action += time.perf_counter() - _t
                n_action_calls += 1

                # Phase 2: env_step
                raw_actions_np = (
                    raw_actions
                    if isinstance(raw_actions, np.ndarray)
                    else raw_actions.cpu().numpy()
                )
                if is_continuous:
                    env_actions = raw_actions_np.reshape(num_envs, 1).astype(np.float32)
                    _t = time.perf_counter()
                    obs_next, rewards, terminations, truncations, info = raw_env.step(
                        (env_actions > 0).astype(np.int32).flatten()
                    )
                    t_env_step += time.perf_counter() - _t
                else:
                    env_actions = raw_actions_np.flatten()
                    _t = time.perf_counter()
                    obs_next, rewards, terminations, truncations, info = raw_env.step(
                        env_actions
                    )
                    t_env_step += time.perf_counter() - _t

                # Convert ONLY what is needed for the next step or agent inference
                obs_next_t = th.as_tensor(obs_next, device=device).unsqueeze(1)

                # Truncation bootstrap obs
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
                    final_obs = final_obs[:, np.newaxis, :]  # [E, A=1, obs_dim]

                # Phase 3: buffer_insert — inputs are [E, A, ...] per [T,E,A,O] spec
                _sync(device)
                _t = time.perf_counter()
                buffer.add(
                    obs=obs_t if full_gpu else obs[:, np.newaxis, :],
                    action=raw_actions if full_gpu else raw_actions_np,
                    reward=rewards[:, np.newaxis],
                    termination=terminations[:, np.newaxis],
                    truncation=truncations[:, np.newaxis],
                    value=values,
                    log_prob=log_probs,
                    final_obs=final_obs,
                )
                _sync(device)
                t_buf_insert += time.perf_counter() - _t

                episode_rewards += rewards
                for i in range(num_envs):
                    if terminations[i] or truncations[i]:
                        rewards_history.append(episode_rewards[i])
                        episode_rewards[i] = 0
                obs = obs_next
                obs_t = obs_next_t
                steps_done += num_envs

            # Phase 4: network_update (PPO)
            agent.train()
            with th.no_grad():
                last_values = agent.expected_V(obs_t)  # [E, A=1]
            _sync(device)
            _t = time.perf_counter()
            agent.reinforcement_learn(
                buffer,
                last_values,
                terminations[:, np.newaxis],  # [E, A=1]
                truncations[:, np.newaxis],  # [E, A=1]
            )
            _sync(device)
            t_net_update += time.perf_counter() - _t
            n_update_calls += 1
            buffer.reset()

        else:
            # --- Off-policy (SAC / DQN): one parallel step, then optional update ---
            # Phase 1: take_action
            _sync(device)
            _t = time.perf_counter()
            with th.no_grad():
                act_dict = agent.train_actions(obs_t)
                raw_actions = act_dict[
                    "continuous_actions" if is_continuous else "discrete_actions"
                ]
            _sync(device)
            t_take_action += time.perf_counter() - _t
            n_action_calls += 1

            # Phase 2: env_step
            raw_actions_np = (
                raw_actions
                if isinstance(raw_actions, np.ndarray)
                else raw_actions.cpu().numpy()
            )
            if is_continuous:
                env_actions = raw_actions_np.reshape(num_envs, 1).astype(np.float32)
                _t = time.perf_counter()
                obs_next, rewards, terminations, truncations, info = raw_env.step(
                    (env_actions > 0).astype(np.int32).flatten()
                )
                t_env_step += time.perf_counter() - _t
            else:
                env_actions = raw_actions_np.flatten()
                _t = time.perf_counter()
                obs_next, rewards, terminations, truncations, info = raw_env.step(
                    env_actions
                )
                t_env_step += time.perf_counter() - _t

            # Convert ONLY what is needed for the next step or agent inference
            obs_next_t = th.as_tensor(obs_next, device=device).unsqueeze(1)

            # Phase 3: buffer_insert — inputs are [E, A, ...] per [T,E,A,O] spec
            _sync(device)
            _t = time.perf_counter()
            buffer.add(
                obs=obs_t if full_gpu else obs[:, np.newaxis, :],
                next_obs=obs_next_t if full_gpu else obs_next[:, np.newaxis, :],
                action=raw_actions if full_gpu else raw_actions_np,
                reward=rewards[:, np.newaxis],
                term=terminations[:, np.newaxis],
                trunc=truncations[:, np.newaxis],
            )
            _sync(device)
            t_buf_insert += time.perf_counter() - _t

            episode_rewards += rewards
            for i in range(num_envs):
                if terminations[i] or truncations[i]:
                    rewards_history.append(episode_rewards[i])
                    episode_rewards[i] = 0
            obs = obs_next
            obs_t = obs_next_t
            steps_done += num_envs

            # Phase 4: network_update (off-policy, scaled)
            if steps_done >= 1000:
                agent.train()
                for _ in range(num_updates_per_step):
                    samples = buffer.sample(batch_size)
                    _sync(device)
                    _t = time.perf_counter()
                    agent.reinforcement_learn(samples, agent_num=0)
                    _sync(device)
                    t_net_update += time.perf_counter() - _t
                    n_update_calls += 1

        if (steps_done - last_print_step) >= 20000:
            last_print_step = steps_done
            avg_rew = np.mean(rewards_history[-100:]) if len(rewards_history) > 0 else 0
            print(f"Step: {steps_done:<8} | Avg Reward: {avg_rew:.2f}")

    end_total = time.perf_counter()
    duration = end_total - start_total
    sps = steps_done / duration

    avg_action_ms = 1000.0 * t_take_action / n_action_calls if n_action_calls else 0.0
    avg_envstep_ms = 1000.0 * t_env_step / n_action_calls if n_action_calls else 0.0
    avg_insert_ms = 1000.0 * t_buf_insert / n_action_calls if n_action_calls else 0.0
    avg_update_ms = 1000.0 * t_net_update / n_update_calls if n_update_calls else 0.0

    print(f"\nPhase Breakdown ({agent_type} {mode} full_gpu={full_gpu}):")
    print(
        f"  take_action  : {t_take_action:8.3f}s total  |  {avg_action_ms:8.4f} ms/call  ({n_action_calls} calls)"
    )
    print(
        f"  env_step     : {t_env_step:8.3f}s total  |  {avg_envstep_ms:8.4f} ms/call"
    )
    print(
        f"  buffer_insert: {t_buf_insert:8.3f}s total  |  {avg_insert_ms:8.4f} ms/call"
    )
    print(
        f"  net_update   : {t_net_update:8.3f}s total  |  {avg_update_ms:8.4f} ms/call  ({n_update_calls} calls)"
    )
    print(f"  Total SPS: {sps:.2f}  |  Duration: {duration:.4f}s")

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
        "full_gpu": full_gpu,
        "sps": sps,
        "duration": duration,
        "t_take_action": t_take_action,
        "t_env_step": t_env_step,
        "t_buf_insert": t_buf_insert,
        "t_net_update": t_net_update,
        "n_action_calls": n_action_calls,
        "n_update_calls": n_update_calls,
        "avg_action_ms": avg_action_ms,
        "avg_envstep_ms": avg_envstep_ms,
        "avg_insert_ms": avg_insert_ms,
        "avg_update_ms": avg_update_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    final_results = []

    algos = ["PPO", "SACV", "SACQ", "DQN"]
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

    print("\n" + "=" * 110)
    print(
        f"{'Agent':<6} {'Mode':<10} {'GPU':<6} {'SPS':>8}  "
        f"{'ActAv(ms)':>10} {'EnvAv(ms)':>10} {'BufAv(ms)':>10} {'UpdAv(ms)':>10}  "
        f"{'ActTot(s)':>9} {'EnvTot(s)':>9} {'BufTot(s)':>9} {'UpdTot(s)':>9}"
    )
    print("-" * 110)
    for r in final_results:
        print(
            f"{r['agent']:<6} {r['mode']:<10} {str(r['full_gpu']):<6} {r['sps']:>8.1f}  "
            f"{r['avg_action_ms']:>10.4f} {r['avg_envstep_ms']:>10.4f} "
            f"{r['avg_insert_ms']:>10.4f} {r['avg_update_ms']:>10.4f}  "
            f"{r['t_take_action']:>9.3f} {r['t_env_step']:>9.3f} "
            f"{r['t_buf_insert']:>9.3f} {r['t_net_update']:>9.3f}"
        )
    print("=" * 110)
