import numpy as np
import torch
from flexibuddiesrl.PG_stabalized import PG
from flexibuddiesrl.Agent import ffEncoder
from itertools import product
import time


def PG_test():
    obs_dim = 12
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
        PG(
            obs_dim=12,
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
        )
        tot += 1
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
