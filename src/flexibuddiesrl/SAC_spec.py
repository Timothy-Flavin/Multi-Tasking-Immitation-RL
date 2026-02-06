import time
import random
from itertools import product

import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:
    gym = None

from flexibuddiesrl.SAC import SAC
from flexibuff import FlexibleBuffer  # , FlexiBatch
import matplotlib.pyplot as plt


def SAC_test():
    obs_dim = 3
    continuous_action_dim = 5
    discrete_action_dims = [3, 5]
    batch_size = 16

    obs = np.random.rand(obs_dim).astype(np.float32)
    obs_batch = np.random.rand(batch_size, obs_dim).astype(np.float32)
    obs_batch_ = obs_batch + 0.1

    # Random discrete actions for two heads
    dacs = np.stack(
        (
            np.random.randint(0, 3, size=(batch_size)),
            np.random.randint(0, 4, size=(batch_size)),
        ),
        axis=-1,
    )

    # Build a buffer matching the SAC.reinforcement_learn expectations
    # Note: Using 'global_reward' (singular) to match SAC implementation
    mem_buff = FlexibleBuffer(
        num_steps=256,
        n_agents=1,
        discrete_action_cardinalities=discrete_action_dims,
        track_action_mask=False,
        path="./test_buffer",
        name="sac_spec_buffer",
        memory_weights=False,
        global_registered_vars={
            "global_rewards": (None, np.float32),
        },
        individual_registered_vars={
            "obs": ([obs_dim], np.float32),
            "obs_": ([obs_dim], np.float32),
            "discrete_actions": ([len(discrete_action_dims)], np.int64),
            "continuous_actions": ([continuous_action_dim], np.float32),
        },
    )

    for i in range(obs_batch.shape[0]):
        c_acs = np.array([-0.5, 0.2, 1.8, 1.9, -2.4], dtype=np.float32)
        mem_buff.save_transition(
            terminated=bool(random.randint(0, 1)),
            registered_vals={
                "global_rewards": i * 1.01,
                "obs": np.array([obs_batch[i]]),
                "obs_": np.array([obs_batch_[i]]),
                "discrete_actions": [dacs[i]],
                "continuous_actions": [c_acs.copy() / (i + 1)],
            },
        )

    param_grid = {
        "discrete_action_dims": [None, [3, 4]],
        "continuous_action_dim": [5, 0],
        "device": ["cuda"],
        "hidden_dims": [[64, 64], [32, 32]],
        "activation": ["tanh", "relu"],
        "orthogonal_init": [False, True],
    }

    p_keys = list(param_grid.keys())
    tot = 0
    for vals in product(*param_grid.values()):
        h = dict(zip(p_keys, vals))
        if h["continuous_action_dim"] == 0 and h["discrete_action_dims"] is None:
            continue
        tot += 1

    start_time = time.time()
    current_time = start_time
    current_iter = 0
    for vals in product(*param_grid.values()):
        h = dict(zip(p_keys, vals))
        if h["continuous_action_dim"] == 0 and h["discrete_action_dims"] is None:
            continue

        t = time.time()
        if t - current_time > 5.0:
            print(
                f"Iter: {current_iter}, time: {(t-start_time):.1f}, iter/s: {current_iter/(t-start_time+1e-9):.1f}, {(current_iter/max(1,tot))*100:.2f}%"
            )
            current_time = t
        current_iter += 1

        dev = h["device"]
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"

        min_actions = (
            np.array([-1, -1, -2, -2, -3], dtype=np.float32)
            if h["continuous_action_dim"] > 0
            else None
        )
        max_actions = (
            np.array([1, 1, 2, 2, 3], dtype=np.float32)
            if h["continuous_action_dim"] > 0
            else None
        )
        print(h)
        model = SAC(
            obs_dim=obs_dim,
            continuous_action_dim=h["continuous_action_dim"],
            discrete_action_dims=h["discrete_action_dims"],
            min_actions=min_actions,
            max_actions=max_actions,
            device=dev,
            hidden_dims=h["hidden_dims"],
            activation=h["activation"],
            orthogonal_init=h["orthogonal_init"],
            lr=1e-3,
        )

        # Single observation actions and expected V
        _ = model.train_actions(obs, step=True, debug=False)
        v = model.expected_V(obs, legal_action=None)
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        # Batch observation actions and expected V
        _ = model.train_actions(obs_batch, step=True, debug=False)
        v_b = model.expected_V(obs_batch, legal_action=None)
        if isinstance(v_b, torch.Tensor):
            v_b = v_b.detach().cpu().numpy()

        # Sample batch and try both imitation and RL
        mb = mem_buff.sample_transitions(
            batch_size=batch_size, as_torch=True, device=dev
        )
        try:
            _ = model.imitation_learn(
                mb.__getattr__("obs")[0],
                mb.__getattr__("continuous_actions")[0],
                mb.__getattr__("discrete_actions")[0],
            )
        except Exception as e:
            msg = f"SAC_spec: imitation_learn failed with: {e}\n"
            msg += f"obs shape: {mb.__getattr__('obs').shape}\n"
            msg += f"ca shape: {mb.__getattr__('continuous_actions').shape}\n"
            msg += f"da shape: {mb.__getattr__('discrete_actions').shape}\n"
            print(msg)
            with open("SAC_log.txt", "a") as f:
                f.write(msg + "\n")
            raise

        try:
            _ = model.reinforcement_learn(mb, 0)
        except Exception as e:
            msg = f"SAC_spec: reinforcement_learn failed with: {e}\n"
            msg += f"config: {h}\n"
            print(msg)
            with open("SAC_log.txt", "a") as f:
                f.write(msg + "\n")
            raise

    print(tot)


def SAC_integration():
    # Simple environment integration similar to DQN/PG integration
    if gym is None:
        print("gymnasium not available; skipping integration test.")
        return

    for config_id in range(4):
        # Alternate between continuous-only and discrete-only setups
        cdim = 2 if config_id % 2 == 0 else 0
        ddim = None if config_id % 2 == 0 else [3, 3]

        mem_buff = FlexibleBuffer(
            num_steps=5000,
            n_agents=1,
            discrete_action_cardinalities=(ddim if ddim is not None else [1]),
            track_action_mask=False,
            path="./test_buffer",
            name="sac_integration_buffer",
            memory_weights=False,
            global_registered_vars={
                "global_rewards": (None, np.float32),
            },
            individual_registered_vars={
                "obs": ([8], np.float32),
                "obs_": ([8], np.float32),
                "discrete_actions": ([len(ddim) if ddim is not None else 1], np.int64),
                "continuous_actions": ([cdim if cdim > 0 else 1], np.float32),
            },
        )
        mem_buff.reset()

        model = SAC(
            obs_dim=8,
            continuous_action_dim=cdim,
            discrete_action_dims=ddim,
            min_actions=(None if cdim == 0 else -np.ones(cdim, dtype=np.float32)),
            max_actions=(None if cdim == 0 else np.ones(cdim, dtype=np.float32)),
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            hidden_dims=[128, 128],
            lr=2e-3,
            actor_every=4,
            initial_temperature=1.0,
            mode="Q",
        )

        gym_env = gym.make("LunarLander-v3", continuous=True)
        obs, _ = gym_env.reset()
        obs_ = obs.copy()
        rewards_hist = [0.0]
        metric_hist = {}
        ep_step = 0

        batch_size = 256
        for i in range(20000):
            with torch.no_grad():
                act = model.train_actions(obs, step=False)
            # Build environment action
            d_act, c_act = 0, 0
            if cdim > 0:
                c_act = act.get("continuous_actions")
                c_np = (
                    c_act.detach().cpu().numpy()
                    if isinstance(c_act, torch.Tensor)
                    else np.asarray(c_act)
                )
                if i < 5000:
                    c_act = np.random.uniform(-1, 1, 2)
                    # print(c_act)

                env_action = np.clip(c_np, -1.0, 1.0)
            else:
                d_act = act.get("discrete_actions")
                env_action = d_act - 1  # type:ignore

                # default_dact = np.array(
                #    d_idx + [0] * (max(1, len(ddim or [])) - len(d_idx)), dtype=np.int64
                # )
                # default_cact = np.zeros((max(1, cdim)), dtype=np.float32)

            obs_, reward, terminated, truncated, _ = gym_env.step(env_action)
            rewards_hist[-1] = rewards_hist[-1] + float(reward)
            mem_buff.save_transition(
                terminated=terminated,
                truncated=truncated,
                registered_vals={
                    "global_rewards": reward,
                    "obs": [obs.copy().astype(np.float32)],
                    "obs_": [obs_.copy().astype(np.float32)],
                    "discrete_actions": d_act,
                    "continuous_actions": c_act,
                },
                bootstrap_values=0.0,
            )

            obs = obs_.copy()
            ep_step += 1
            if terminated or truncated:
                rewards_hist.append(0.0)
                if len(rewards_hist) % 16 == 15:
                    gym_env = gym.make(
                        "LunarLander-v3", continuous=True, render_mode="human"
                    )
                else:
                    gym_env = gym.make(
                        "LunarLander-v3",
                        continuous=True,
                    )
                obs, _ = gym_env.reset()
                ep_step = 0
                print(f"reward: {rewards_hist[-2]} ep: {len(rewards_hist)} step: {i}")
            if mem_buff.steps_recorded > batch_size * 4:
                mb = mem_buff.sample_transitions(
                    batch_size=batch_size, as_torch=True, device=model.device
                )
                rl_metrics = model.reinforcement_learn(mb, 0, critic_only=i < 5000)
                for k in rl_metrics.keys():
                    if k not in metric_hist:
                        metric_hist[k] = [rl_metrics[k]]
                    else:
                        if rl_metrics[k] == 0.0:
                            rl_metrics[k] = metric_hist[k][-1]
                        else:
                            metric_hist[k].append(rl_metrics[k])
        plt.plot(rewards_hist)
        plt.title("Reward hist")
        plt.grid()
        plt.tight_layout()
        plt.show()

        maxes = {}
        names = []
        for k in metric_hist.keys():
            metric_hist[k] = np.array(metric_hist[k])
            maxes[k] = max(np.max(np.abs(metric_hist[k])), 0.0001)
            metric_hist[k] /= maxes[k]
            plt.plot(metric_hist[k])
            names.append(k + ", " + str(maxes[k]))
        plt.legend(names)
        plt.grid()
        plt.tight_layout()
        plt.title("RL metrics tracking")
        plt.show()
        print(model)


if __name__ == "__main__":
    SAC_test()
    SAC_integration()
