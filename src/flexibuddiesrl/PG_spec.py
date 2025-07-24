import numpy as np
import torch
from flexibuddiesrl.PG_stabalized import PG
from flexibuddiesrl.Agent import ffEncoder
from itertools import product
import time
from flexibuff import FlexibleBuffer
import random


def PG_test():
    obs_dim = 3
    continuous_action_dim = 2
    discrete_action_dims = [4, 5]
    obs = np.random.rand(obs_dim).astype(np.float32)
    obs_ = np.random.rand(obs_dim).astype(np.float32)
    obs_batch = np.random.rand(14, obs_dim).astype(np.float32)
    obs_batch_ = obs_batch + 0.1

    dacs = np.stack(
        (np.random.randint(0, 4, size=(14)), np.random.randint(0, 5, size=(14))),
        axis=-1,
    )

    mem_buff = FlexibleBuffer(
        num_steps=64,
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
        c_acs = np.arange(0, continuous_action_dim, dtype=np.float32)
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
                "continuous_actions": [c_acs.copy() + i / obs_batch.shape[0]],
            },
        )
    mem = mem_buff.sample_transitions(batch_size=14, as_torch=True, device="cuda")

    param_grid = {
        "continuous_action_dim": [0, 5],
        "discrete_action_dims": [None, [3, 4]],
        "device": ["cpu", "cuda"],
        "entropy_loss": [0, 0.05],
        "ppo_clip": (0, 0.2),
        "value_clip": (0, 0.5),
        "norm_advantages": (False, True),
        "anneal_lr": (0, 20000),
        "orthogonal": (True, False),
        "std_type": ["full", "diagonal", "stateless"],
        "clip_grad": (True, False),
        "eval_mode": (False, True),
        "action_head_hidden_dims": (None, [32, 32]),
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
        if t - current_time > 1.0:
            print(
                f"Iter: {current_iter}, time: {(t-start_time):.1f}, iter/s: {current_iter/(t-start_time):.1f}, {(current_iter/tot)*100:.2f}%"
            )
            current_time = t
        current_iter += 1
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
        )
        d_acts, c_acts, d_log, c_log, _ = model.train_actions(
            obs, step=True, debug=False
        )
        if (d_acts is not None and d_acts.shape[0] != 2) or (
            c_acts is not None and c_acts.shape != 5
        ):
            print(
                f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}"
            )

    print(tot)
    # for dev in device:
    #     temp_enc = ffEncoder(12, [32, 32], device=dev)
    #     hidden_encoder = [[[32, 32], None], [None, temp_enc]]
    #     for h_sizes, enc in hidden_encoder:
    #         for con in continuous_action_dim:
    #             for dis in discrete_action_dims:
    #                 for eloss in entropy_loss:
    #                     for ppc in ppo_clip:
    #                         for vclip in value_clip:
    #                             for n_adv in norm_advantages:
    #                                 for lra in anneal_lr:
    #                                     for ortho in orthogonal:
    #                                         for stdt in std_type:
    #                                             for clipg in clip_grad:
    #                                                 for eval in eval_mode:
    #                                                     for (
    #                                                         ahd
    #                                                     ) in action_head_hidden_dims:
    #                                                         tot += 1
    # print(tot)
    # PG(
    #     obs_dim=obs_dim,
    #     continuous_action_dim=con,
    #     min_actions=(np.array([-1, -1, -2, -2, -3]) if con > 0 else np.zeros(1)),
    #     max_actions=(np.array([1, 1, 2, 2, 3]) if con > 0 else np.zeros(1)),
    # )


if __name__ == "__main__":
    PG_test()
