import numpy as np
import torch
from flexibuddiesrl.Agent import QS


def QS_test():
    mat = torch.from_numpy(np.random.rand(32, 12))

    duel_tests = [True, False]
    dis_tests = [None, [2, 3, 4]]
    con_tests = [0, 5]
    head_hidden_tests = [None, 64]

    for duel in duel_tests:
        for dis in dis_tests:
            for con in con_tests:
                for head_hidden in head_hidden_tests:
                    print(
                        f"Testing with dueling={duel}, discrete={dis}, continuous={con}, head_hidden={head_hidden}"
                    )
                    Q = QS(
                        obs_dim=12,
                        continuous_action_dim=con,
                        discrete_action_dims=dis,
                        hidden_dims=[32, 32],
                        dueling=duel,
                        n_c_action_bins=3,
                        head_hidden_dim=head_hidden,
                    )
                    v, d, c = Q(mat)
                    if duel:
                        print("  Value shape:", v.shape)
                    if dis is not None:
                        print("  Discrete action dimensions:", len(d))
                        for dim in d:
                            print("    Discrete action dim shape:", dim.shape)
                    if con > 0:
                        print("  Continuous action shape:", c.shape)


if __name__ == "__main__":
    QS_test()
