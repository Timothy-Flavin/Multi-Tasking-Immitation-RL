import numpy as np
import torch
from flexibuddiesrl.DQN import DQN
from flexibuddiesrl.Agent import ffEncoder
from itertools import product
import time
from flexibuff import FlexibleBuffer, FlexiBatch
import random
import time
import gymnasium as gym
import matplotlib.pyplot as plt


def DQN_test():
    obs_dim = 3
    continuous_action_dim = 5
    discrete_action_dims = [3, 5]
    batch_size = 16
    mini_batch_size = 8
    obs = np.random.rand(obs_dim).astype(np.float32)
    obs_ = np.random.rand(obs_dim).astype(np.float32)
    obs_batch = np.random.rand(batch_size, obs_dim).astype(np.float32)
    obs_batch_ = obs_batch + 0.1

    run_times = {
        "create_model": 0.0,
        "train_action_single": 0.0,
        "train_action_batch": 0.0,
        "imitation_learn": 0.0,
        "reinforcement_learn": 0.0,
    }

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
            # "discrete_log_probs": ([len(discrete_action_dims)], np.float32),
            # "continuous_log_probs": (None, np.float32),
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
                # "discrete_log_probs": np.zeros(
                #    len(discrete_action_dims), dtype=np.float32
                # )
                # - i / obs_batch.shape[0]
                # - 0.1,
                # "continuous_log_probs": np.zeros(1, dtype=np.float32)
                # - i / obs_batch.shape[0] / 2
                # - 0.1,
                "discrete_actions": [dacs[i]],
                "continuous_actions": [c_acs.copy() / (i + 1)],
            },
        )

    param_grid = {
        "discrete_action_dims": [None, [3, 4]],  # np.array([2]),
        "continuous_action_dim": [5, 0],  # 2,
        "head_hidden_dim": [None, [32]],  # if None then no head hidden layer
        "dueling": [True, False],
        "munchausen": [0.0, 0.9],  # turns it into munchausen dqn
        "entropy": [0.0, 0.1],  # turns it into soft-dqn
        "device": ["cpu", "cuda"],
        "conservative": [False, True],
        "imitation_type": ["cross_entropy", "reward"],  # or "reward"
    }
    p_keys = param_grid.keys()
    tot = 0
    for vals in product(*param_grid.values()):
        h = dict(zip(p_keys, vals))
        if h["continuous_action_dim"] == 0 and h["discrete_action_dims"] is None:
            continue
        if h["munchausen"] > 0.0 and h["entropy"] == 0.0:
            continue
        tot += 1
    # print(tot)
    start_time = time.time()
    current_time = time.time()
    current_iter = 0
    for vals in product(*param_grid.values()):
        h = dict(zip(p_keys, vals))
        print(h)
        if h["continuous_action_dim"] == 0 and h["discrete_action_dims"] is None:
            continue
        if h["munchausen"] > 0.0 and h["entropy"] == 0.0:
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

            # rl_tot = 0.0
            # for k in rl_times:
            #     if k != "tot":
            #         rl_tot += rl_times[k]
            # print(f"     Captured: {rl_tot/rl_times['tot'] *100:.3f}%")
            # for k in rl_times:
            #     if k != "tot":
            #         print(f"     {k}: {rl_times[k] / rl_times['tot'] *100:.2f}%")

            current_time = t
        current_iter += 1

        _s = time.time()
        model = DQN(
            obs_dim=obs_dim,
            discrete_action_dims=h["discrete_action_dims"],
            continuous_action_dims=h["continuous_action_dim"],
            min_actions=(
                None if h["continuous_action_dim"] == 0 else np.zeros(5)
            ),  # np.array([-1,-1]),
            max_actions=(
                None if h["continuous_action_dim"] == 0 else np.ones(5)
            ),  # ,np.array([1,1]),
            hidden_dims=[64, 64],  # first is obs dim if encoder provded
            head_hidden_dim=h["head_hidden_dim"],  # if None then no head hidden layer
            gamma=0.99,
            lr=3e-5,
            imitation_lr=1e-5,
            dueling=h["dueling"],
            n_c_action_bins=5,
            munchausen=h["munchausen"],  # turns it into munchausen dqn
            entropy=h["entropy"],  # turns it into soft-dqn
            activation="relu",
            orthogonal=False,
            init_eps=0.9,
            eps_decay_half_life=10000,
            device=h["device"],
            eval_mode=False,
            name="DQN",
            clip_grad=1.0,
            load_from_checkpoint_path=None,
            encoder=None,
            conservative=h["conservative"],
            imitation_type=h["imitation_type"],  # or "reward"
        )
        run_times["create_model"] += time.time() - _s

        _s = time.time()
        d_acts, c_acts, d_log, c_log, _1, _ = model.train_actions(
            obs, step=True, debug=False
        )
        run_times["train_action_single"] += time.time() - _s

        # if (d_acts is not None and d_acts.shape[0] != 2) or (
        #     c_acts is not None and c_acts.shape[0] != 5
        # ):
        #     print(
        #         f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}"
        #     )

        _s = time.time()
        d_acts, c_acts, d_log, c_log, _1, _ = model.train_actions(
            obs_batch, step=True, debug=False
        )
        run_times["train_action_batch"] += time.time() - _s

        # if (
        #     d_acts is not None
        #     and (d_acts.shape[0] != batch_size or d_acts.shape[1] != 2)
        # ) or (
        #     c_acts is not None
        #     and (c_acts.shape[0] != batch_size or c_acts.shape[1] != 5)
        # ):
        #     print(
        #         f"Training batch actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}"
        #     )
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
            print("Couldn't imitation learn ")
            print(mb.__getattr__("obs"))
            print(
                f"obs: {mb.__getattr__('obs')}, ca: {mb.__getattr__('continuous_actions')}, da: {mb.__getattr__('discrete_actions')}"
            )
            print(h)
            raise e
        run_times["imitation_learn"] += time.time() - _s

        _s = time.time()
        try:
            aloss, closs = model.reinforcement_learn(mb, 0)
        except Exception as e:
            print(h)
            raise e
        run_times["reinforcement_learn"] += time.time() - _s
        print(f"time: {time.time()-t}")
    print(tot)


if __name__ == "__main__":
    DQN_test()
