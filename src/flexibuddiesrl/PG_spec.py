import numpy as np
import torch
from flexibuddiesrl.PG_stabalized import PG
from flexibuddiesrl.Agent import ffEncoder
from itertools import product
import time
from flexibuff import FlexibleBuffer, FlexiBatch
import random
import time
import gymnasium as gym


def PG_test():
    run_times = {
        "create_model": 0.0,
        "train_action_single": 0.0,
        "train_action_batch": 0.0,
        "immitation_learn": 0.0,
        "reinforcement_learn": 0.0,
    }
    rl_times = {
        "advantage": 0.0,
        "critic_loss": 0.0,
        "act": 0.0,
        "dloss": 0.0,
        "closs": 0.0,
        "backward": 0.0,
        "tot": 0.0,
    }
    obs_dim = 3
    continuous_action_dim = 5
    discrete_action_dims = [3, 5]
    batch_size = 16
    mini_batch_size = 8
    obs = np.random.rand(obs_dim).astype(np.float32)
    obs_ = np.random.rand(obs_dim).astype(np.float32)
    obs_batch = np.random.rand(batch_size, obs_dim).astype(np.float32)
    obs_batch_ = obs_batch + 0.1

    dacs = np.stack(
        (
            np.random.randint(0, 3, size=(batch_size)),
            np.random.randint(0, 4, size=(batch_size)),
        ),
        axis=-1,
    )

    mem_buff = FlexibleBuffer(
        num_steps=256,
        n_agents=1,
        discrete_action_cardinalities=discrete_action_dims,
        track_action_mask=False,
        path="./test_buffer",
        name="spec_buffer",
        memory_weights=False,
        global_registered_vars={
            "global_rewards": (None, np.float32),
        },
        individual_registered_vars={
            "obs": ([obs_dim], np.float32),
            "obs_": ([obs_dim], np.float32),
            "discrete_log_probs": ([len(discrete_action_dims)], np.float32),
            "continuous_log_probs": ([continuous_action_dim], np.float32),
            "discrete_actions": ([len(discrete_action_dims)], np.int64),
            "continuous_actions": ([continuous_action_dim], np.float32),
        },
    )
    for i in range(obs_batch.shape[0]):
        c_acs = np.array(
            [-0.5, 0.2, 1.8, 1.9, -2.4], dtype=np.float32
        )  # np.arange(0, continuous_action_dim, dtype=np.float32)
        mem_buff.save_transition(
            terminated=bool(random.randint(0, 1)),
            registered_vals={
                "global_rewards": i * 1.01,
                "obs": np.array([obs_batch[i]]),
                "obs_": np.array([obs_batch_[i]]),
                "discrete_log_probs": np.zeros(
                    len(discrete_action_dims), dtype=np.float32
                )
                - i / obs_batch.shape[0]
                - 0.1,
                "continuous_log_probs": np.zeros(
                    continuous_action_dim, dtype=np.float32
                )
                - i / obs_batch.shape[0] / 2
                - 0.1,
                "discrete_actions": [dacs[i]],
                "continuous_actions": [c_acs.copy() / (i + 1)],
            },
        )

    param_grid = {
        "action_clamp_type": ["tanh", "clamp", None],
        "continuous_action_dim": [5, 0],
        "discrete_action_dims": [[3, 4], None],
        "device": ["cuda", "cpu"],
        "std_type": ["stateless", "diagonal", "full"],
        "entropy_loss": [0, 0.05],
        "ppo_clip": (0, 0.2),
        "value_clip": (0, 0.5),
        "norm_advantages": (False, True),
        "anneal_lr": (0, 20000),
        "orthogonal": (True, False),
        "clip_grad": (False, True),
        "eval_mode": (False, True),
        "action_head_hidden_dims": (None, [8, 4]),
        "adv_type": ["gae", "gv", "a2c", "g", "constant"],
    }

    p_keys = param_grid.keys()
    tot = 0
    for vals in product(*param_grid.values()):
        h = dict(zip(p_keys, vals))
        if h["continuous_action_dim"] == 0 and h["discrete_action_dims"] is None:
            continue
        tot += 1
    print(tot)
    start_time = time.time()
    current_time = time.time()
    current_iter = 0
    for vals in product(*param_grid.values()):
        h = dict(zip(p_keys, vals))
        if h["continuous_action_dim"] == 0 and h["discrete_action_dims"] is None:
            continue
        # print(h)
        t = time.time()
        if t - current_time > 5.0:
            print(
                f"Iter: {current_iter}, time: {(t-start_time):.1f}, iter/s: {current_iter/(t-start_time):.1f}, {(current_iter/tot)*100:.2f}%"
            )
            tot_t = 0.0
            for k in run_times.keys():
                tot_t += run_times[k]
            for k in run_times.keys():
                print(f"  {k}: {run_times[k] / tot_t *100:.2f}%")

            rl_tot = 0.0
            for k in rl_times:
                if k != "tot":
                    rl_tot += rl_times[k]
            print(f"     Captured: {rl_tot/rl_times['tot'] *100:.3f}%")
            for k in rl_times:
                if k != "tot":
                    print(f"     {k}: {rl_times[k] / rl_times['tot'] *100:.2f}%")

            current_time = t
        current_iter += 1

        _s = time.time()
        model = PG(
            obs_dim=obs_dim,
            continuous_action_dim=h["continuous_action_dim"],
            discrete_action_dims=h["discrete_action_dims"],
            min_actions=(
                np.array([-1, -1, -2, -2, -3])
                if h["continuous_action_dim"] > 0
                else np.zeros(1)
            ),
            max_actions=(
                np.array([1, 1, 2, 2, 3])
                if h["continuous_action_dim"] > 0
                else np.zeros(1)
            ),
            device=h["device"],
            entropy_loss=h["entropy_loss"],
            ppo_clip=h["ppo_clip"],
            value_clip=h["value_clip"],
            norm_advantages=h["norm_advantages"],
            anneal_lr=h["anneal_lr"],
            orthogonal=h["orthogonal"],
            std_type=h["std_type"],
            clip_grad=h["clip_grad"],
            mini_batch_size=mini_batch_size,
            action_clamp_type=h["action_clamp_type"],
            advantage_type=h["adv_type"],
            n_epochs=1,
        )
        run_times["create_model"] += time.time() - _s

        _s = time.time()
        d_acts, c_acts, d_log, c_log, _ = model.train_actions(
            obs, step=True, debug=False
        )
        run_times["train_action_single"] += time.time() - _s

        if (d_acts is not None and d_acts.shape[0] != 2) or (
            c_acts is not None and c_acts.shape[0] != 5
        ):
            print(
                f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}"
            )

        _s = time.time()
        d_acts, c_acts, d_log, c_log, _ = model.train_actions(
            obs_batch, step=True, debug=False
        )
        run_times["train_action_batch"] += time.time() - _s

        if (
            d_acts is not None
            and (d_acts.shape[0] != batch_size or d_acts.shape[1] != 2)
        ) or (
            c_acts is not None
            and (c_acts.shape[0] != batch_size or c_acts.shape[1] != 5)
        ):
            print(
                f"Training batch actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}"
            )
        mb = mem_buff.sample_transitions(
            batch_size=batch_size, as_torch=True, device=h["device"]
        )
        # print(mb)

        _s = time.time()
        try:
            aloss, closs = model.imitation_learn(
                mb.__getattr__("obs")[0],
                mb.__getattr__("continuous_actions")[0],
                mb.__getattr__("discrete_actions")[0],
            )
        except Exception as e:
            print("Couldn't immitation learn ")
            print(mb.__getattr__("obs"))
            print(
                f"obs: {mb.__getattr__('obs')}, ca: {mb.__getattr__('continuous_actions')}, da: {mb.__getattr__('discrete_actions')}"
            )
            print(h)
            raise e
        run_times["immitation_learn"] += time.time() - _s

        _s = time.time()
        try:
            aloss, closs = model.reinforcement_learn_perf(mb, 0)
        except Exception as e:
            print(h)
            raise e
        run_times["reinforcement_learn"] += time.time() - _s

        for k in rl_times:
            rl_times[k] += model.run_times[k]

    print(tot)


def PG_integration():

    gym.make("CartPole-v1")

    mem_buff = FlexibleBuffer(
        num_steps=10000,
        n_agents=1,
        discrete_action_cardinalities=[2],
        track_action_mask=False,
        path="./test_buffer",
        name="spec_buffer",
        memory_weights=False,
        global_registered_vars={
            "global_rewards": (None, np.float32),
        },
        individual_registered_vars={
            "obs": ([4], np.float32),
            "obs_": ([4], np.float32),
            "discrete_log_probs": ([1], np.float32),
            "continuous_log_probs": (None, np.float32),
            "discrete_actions": ([1], np.int64),
            "continuous_actions": ([1], np.float32),
        },
    )
    param_grid = {
        "action_clamp_type": ["tanh", "clamp", None],
        "continuous_action_dim": [5, 0],
        "discrete_action_dims": [[3, 4], None],
        "device": ["cuda", "cpu"],
        "std_type": ["stateless", "diagonal", "full"],
        "entropy_loss": [0, 0.05],
        "ppo_clip": (0, 0.2),
        "value_clip": (0, 0.5),
        "norm_advantages": (False, True),
        "anneal_lr": (0, 20000),
        "orthogonal": (True, False),
        "clip_grad": (False, True),
        "eval_mode": (False, True),
        "action_head_hidden_dims": (None, [8, 4]),
        "adv_type": ["gae", "gv", "a2c", "g", "constant"],
    }

    for config_id in range(10):
        mem_buff.reset()
        cdim = 0
        ddim = None
        if config_id % 2 == 0:
            cdim = 1
        else:
            ddim = [2]

        model = PG(
            obs_dim=4,
            continuous_action_dim=cdim,
            discrete_action_dims=ddim,
            min_actions=(np.array([0]) if cdim > 0 else np.zeros(1)),
            max_actions=(np.array([1]) if cdim > 0 else np.zeros(1)),
            device=param_grid["device"][random.randint(0, 1)],
            entropy_loss=param_grid["entropy_loss"][random.randint(0, 1)],
            ppo_clip=param_grid["ppo_clip"][random.randint(0, 1)],
            value_clip=param_grid["value_clip"][random.randint(0, 1)],
            norm_advantages=param_grid["norm_advantages"][random.randint(0, 1)],
            anneal_lr=param_grid["anneal_lr"][random.randint(0, 1)],
            orthogonal=param_grid["orthogonal"][random.randint(0, 1)],
            std_type=param_grid["std_type"][random.randint(0, 2)],
            clip_grad=param_grid["clip_grad"][random.randint(0, 1)],
            mini_batch_size=16,
            action_clamp_type=param_grid["action_clamp_type"][random.randint(0, 2)],
            advantage_type=param_grid["adv_type"][random.randint(0, 4)],
            n_epochs=2,
        )

        gym_env = gym.make("CartPole-v1")
        obs, _ = gym_env.reset()
        obs_ = obs + 0.1
        batch_size = 64
        rewards = [0.0]
        ep_num = 0

        for i in range(5000):
            with torch.no_grad():
                env_action = 0
                default_dact = np.zeros((1), dtype=np.int64)
                default_cact = np.zeros((1), dtype=np.float32) - 0.5
                default_clp = np.ones((1), dtype=np.float32)
                default_dlp = np.ones((1, 1), dtype=np.float32)
                dact, cact, dlp, clp, v = model.train_actions(
                    obs, step=True, debug=False
                )

            if cdim > 0:
                assert (
                    cact is not None and clp is not None
                ), f"Continuous action and log prob {cact} {clp} should not be None when cdim [{cdim}] is not 0"
                print(f"Continuous action: {cact}, log prob: {clp}")
                # print(cact.shape, clp.shape)
                # print("from logits look like:")
                # print(model.actor.forward(obs))
                # print(model.action_clamp_type)

                env_action = int(cact[0] > 0.5)
                default_cact[0] = cact[0]
                default_clp[0] = clp
            else:
                assert (
                    dact is not None and dlp is not None
                ), f"Discrete action and log prob {dact} {dlp} should not be None when cdim [{cdim}] is 0"
                print(dact.shape, dlp.shape, dact, dlp)
                env_action = dact[0]
                default_dact[0] = dact[0]
                default_dlp[0][0] = dlp[0]

            obs_, reward, terminated, truncated, _ = gym_env.step(env_action)
            rewards[-1] = rewards[-1] + float(reward)
            rv = {
                "global_rewards": reward,
                "obs": [obs.copy()],
                "obs_": [obs_.copy()],
                "discrete_log_probs": default_dlp.copy(),
                "continuous_log_probs": default_clp.copy(),
                "discrete_actions": default_dact.copy(),
                "continuous_actions": default_cact.copy(),
            }
            # print(rv)
            mem_buff.save_transition(
                terminated=terminated,
                registered_vals=rv,
            )

            obs = obs_.copy()
            if terminated or truncated:
                obs, _ = gym_env.reset()
                obs = obs.copy()
                rewards.append(0.0)
                print(f"Episode {ep_num}, total reward: {rewards[-2]}")
                ep_num += 1

            if mem_buff.steps_recorded > batch_size:
                print(model.action_clamp_type)
                mb = mem_buff.sample_transitions(
                    idx=np.arange(0, batch_size), as_torch=True, device=model.device
                )
                aloss, closs = model.reinforcement_learn(mb, 0)
                print(f"Iteration {i}, aloss: {aloss}, closs: {closs}")
                mem_buff.reset()


if __name__ == "__main__":
    PG_integration()
    PG_test()
