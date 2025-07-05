# %%
import numpy as np
import torch
import flexibuddiesrl
from flexibuddiesrl.Agent import QS, StochasticActor


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


def SA_test():
    mat = torch.from_numpy(np.random.rand(32, 12))

    dis_tests = [None, [3, 4]]
    con_tests = [0, 5]
    head_hidden_tests = [None, 64]

    for dis in dis_tests:
        for con in con_tests:
            for head_hidden in head_hidden_tests:
                print(
                    f"Testing with discrete={dis}, continuous={con}, head_hidden={head_hidden}"
                )
                sa = StochasticActor(
                    obs_dim=12,
                    continuous_action_dim=con,
                    discrete_action_dims=dis,
                    hidden_dims=np.array([32, 32]),
                    action_head_hidden_dims=head_hidden,
                )
                ca, calp, da = sa(mat)
                if da is not None:
                    print("  Discrete log prob heads:", len(da))
                    for d in da:
                        print("    Discrete action dim shape:", d.shape)
                # if con > 0:
                #    print("  Continuous action shape:", c.shape)


# %%
if __name__ == "__main__":
    data = torch.from_numpy(np.array([[0.0, 1.1, -1.1, 2.0], [0.1, 1.2, -1.3, 2.4]]))
    sa = StochasticActor(
        obs_dim=4,
        continuous_action_dim=0,
        max_actions=np.array([2, 2, 2, 3]),
        min_actions=np.array([-1, -1, -1, -3]),
        discrete_action_dims=[2, 3],
        hidden_dims=np.array([16, 32]),
        action_head_hidden_dims=np.array([48]),
        std_type="full",
    )
    print(
        sa(
            data[0],
        )
    )
    # QS_test()
    # SA_test()

# %%
