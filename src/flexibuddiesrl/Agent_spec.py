# %%
import numpy as np
import torch
from flexibuddiesrl.Agent import QS, StochasticActor, ffEncoder


def QS_test(verbose=False):
    mat = torch.from_numpy(np.random.rand(18, 12))
    encoder = ffEncoder(12, [24, 12])

    duel_tests = [True, False]
    value_per_head_tests = [True, False]
    dis_tests = [None, [2, 3]]
    con_tests = [0, 5]
    head_hidden_tests = [None, [32], [16, 16]]
    encoder_arg = [None, encoder]

    total_tests = 0
    ca_passes = 0
    da_passes = 0
    v_passes = 0
    for duel in duel_tests:
        for sv in value_per_head_tests:
            for dis in dis_tests:
                for con in con_tests:
                    if dis is None and con == 0:
                        continue
                    for head_hidden in head_hidden_tests:
                        for enc in encoder_arg:
                            total_tests += 1
                            if verbose:
                                print(
                                    f"Testing with dueling={duel}, discrete={dis}, continuous={con}, head_hidden={head_hidden}, encoder: {enc is not None}"
                                )
                            Q = QS(
                                obs_dim=12,
                                continuous_action_dim=con,
                                discrete_action_dims=dis,
                                hidden_dims=[32, 32],
                                dueling=duel,
                                n_c_action_bins=3,
                                head_hidden_dims=(
                                    np.array(head_hidden)
                                    if head_hidden is not None
                                    else None
                                ),
                                encoder=enc,
                                value_per_head=sv,
                            )
                            bv, bd, bc = Q(mat)
                            v, d, c = Q(mat[0])
                            con_pass = True
                            dis_pass = True
                            val_pass = True

                            # batch passing
                            if con > 0:
                                if bc.shape != torch.Size(
                                    (18, con, 3)
                                ) or c.shape != torch.Size([con, 3]):
                                    con_pass = False
                                    input(
                                        f"Failed Testing with dueling={duel}, discrete={dis}, continuous={con}, head_hidden={head_hidden}, encoder: {enc is not None} shared: {sv} + d: {c}, vb:[{bc.shape},{c.shape}]"
                                    )

                            if dis is not None:
                                if (
                                    len(bd) != 2
                                    or bd[0].shape != torch.Size((18, 2))
                                    or bd[1].shape != torch.Size((18, 3))
                                ):
                                    dis_pass = False
                                if (
                                    len(d) != 2
                                    or d[0].shape != torch.Size([2])
                                    or d[1].shape != torch.Size([3])
                                ):
                                    dis_pass = False
                                    # print(bd[0])
                                    input(
                                        f"Failed Testing with dueling={duel}, discrete={dis}, continuous={con}, head_hidden={head_hidden}, encoder: {enc is not None} shared: {sv} + d: {d}, vb:[{bd[0].shape},{bd[1].shape}]"
                                    )

                            dis_dim = 0 if dis is None else 2

                            v_size = dis_dim + con if sv else 1
                            if duel and (
                                bv.shape != torch.Size([18, v_size])
                                or v.shape != torch.Size([v_size])
                            ):
                                v_passes = False
                                input(
                                    f"Failed Testing with dueling={duel}, discrete={dis}, continuous={con}, head_hidden={head_hidden}, encoder: {enc is not None} shared: {sv} vsize: {(v_size)} + v: {v.shape}, vb: {bv.shape}"
                                )
                            # single action passing
                            ca_passes += con_pass
                            da_passes += dis_pass
                            v_passes += val_pass
                            if verbose:
                                if duel:
                                    print("  Value shape:", v.shape)
                                if dis is not None:
                                    print("  Discrete action dimensions:", len(d))
                                    for dim in d:
                                        print(
                                            "    Discrete action dim shape:", dim.shape
                                        )
                                if con > 0:
                                    print("  Continuous action shape:", c.shape)
    print(f"Total tests run: {total_tests}")
    print(f"value passes: {v_passes}")
    print(f"continuous action passes: {ca_passes}")
    print(f"discrete action passes: {da_passes}")


def SA_test(verbose=False):
    mat = torch.from_numpy(np.random.rand(18, 12)).float()

    encoder = ffEncoder(12, [24, 12]).float()
    dis_tests = [None, [3, 4]]
    con_tests = [0, 5]
    head_hidden_tests = [None, [64]]
    log_types = ["full", "diagonal", None, "none"]
    encoder_test = [None, encoder]
    gumble_test = [True, False]

    total_tests = 0
    ca_pass_count = 0
    log_pass_count = 0
    da_pass_count = 0

    for dis in dis_tests:
        for con in con_tests:
            if con == 0 and dis is None:
                continue
            for head_hidden in head_hidden_tests:
                for enc in encoder_test:
                    for lstd in log_types:
                        for gum in gumble_test:

                            total_tests += 1
                            print(
                                f"Testing with discrete={dis}, continuous={con}, head_hidden={head_hidden}, encoder: {enc is not None}, logtype: {lstd}, gumble: {gum}"
                            )
                            sa = StochasticActor(
                                obs_dim=12,
                                continuous_action_dim=con,
                                discrete_action_dims=dis,
                                min_actions=(
                                    np.array([-1, -1, -2, -2, -3])
                                    if con > 0
                                    else np.zeros(1)
                                ),
                                max_actions=(
                                    np.array([1, 1, 2, 2, 3])
                                    if con > 0
                                    else np.zeros(1)
                                ),
                                hidden_dims=np.array([32, 32]),
                                action_head_hidden_dims=head_hidden,
                                encoder=enc,
                                std_type=lstd,
                            )
                            ca, calp, da = sa(mat)
                            if verbose:
                                print("  testing for batch action to logits")
                                if da is not None:
                                    print("  Discrete log prob heads:", len(da))
                                    for d in da:
                                        print("    Discrete action dim shape:", d.shape)
                                if con > 0:
                                    print("  Continuous action shape:", ca.shape)
                                    print(f"  Continuous lp shape: {calp.shape}")

                            ca_passing = True
                            da_passing = True
                            lstd_passing = True
                            if verbose:
                                print("  Testing batch action logits")
                            if con > 0:
                                if ca.shape[0] != 18 or ca.shape[1] != con:
                                    ca_passing = False
                                if (
                                    lstd is not None
                                    and lstd != "none"
                                    and calp.shape[0] != 18
                                ):
                                    lstd_passing = False
                                if lstd == "full" and (
                                    calp.shape[0] != 18 or calp.shape[1] != con
                                ):
                                    lstd_passing = False
                                if lstd == "diagonal" and (
                                    calp.shape[0] != 18 or calp.shape[1] != 1
                                ):
                                    lstd_passing = False
                            if dis is not None:
                                if da[0].shape[0] != 18 or da[0].shape[1] != 3:
                                    da_passing = False
                                if da[1].shape[0] != 18 or da[1].shape[1] != 4:
                                    da_passing = False
                            if verbose:
                                print(
                                    f"  Batch logits passing da {da_passing} ca {ca_passing} lp {lstd_passing}"
                                )
                                print("  Testing single action ")
                            ca, calp, da = sa(mat[0])

                            if con > 0:
                                if ca.shape[0] != con or len(ca.shape) != 1:
                                    ca_passing = False
                                if (
                                    lstd is not None
                                    and lstd != "none"
                                    and len(calp.shape) != 1
                                ):
                                    lstd_passing = False
                                if lstd == "full" and (
                                    calp.shape[0] != con or len(calp.shape) != 1
                                ):
                                    lstd_passing = False
                                if lstd == "diagonal" and (
                                    calp.shape[0] != 1 or len(calp.shape) != 1
                                ):
                                    lstd_passing = False
                            if dis is not None:
                                if da[0].shape[0] != 3 or len(da[0].shape) != 1:
                                    da_passing = False
                                if da[1].shape[0] != 4 or len(da[1].shape) != 1:
                                    da_passing = False
                            if verbose:
                                print(
                                    f"  Single logits passing da {da_passing} ca {ca_passing} lp {lstd_passing}"
                                )
                                print("Testing action sampler")

                            batch_ca, batch_calp, batch_da = sa(mat)
                            batch_sample_ca, batch_sample_da = sa.action_from_logits(
                                batch_ca, batch_calp, batch_da, gumble=gum
                            )
                            sample_ca, sample_da = sa.action_from_logits(
                                ca, calp, da, gumble=gum
                            )

                            if con > 0:
                                if (
                                    batch_sample_ca.shape[0] != 18  # type:ignore
                                    or batch_sample_ca.shape[1] != con  # type:ignore
                                ):
                                    ca_passing = False
                            if dis is not None:
                                if gum:
                                    if (
                                        batch_da[0].shape[0] != 18
                                        or batch_da[1].shape[0] != 18
                                        or batch_da[0].shape[1] != 3
                                        or batch_da[1].shape[1] != 4
                                    ):
                                        da_passing = False
                                else:
                                    assert batch_sample_da is not None
                                    assert isinstance(batch_sample_da, torch.Tensor)
                                    if batch_sample_da.shape[
                                        0
                                    ] != 18 or batch_sample_da.shape[1] != len(dis):
                                        da_passing = False
                            da_pass_count += da_passing
                            ca_pass_count += ca_passing
                            log_pass_count += lstd_passing

                            if not da_passing or not ca_passing or not lstd_passing:
                                print(
                                    f"One or more tests failed for with discrete={dis}, continuous={con}, head_hidden={head_hidden}, encoder: {enc is not None}, logtype: {lstd}, gumble {gum}"
                                )
                            if not da_passing:
                                print("  Discrete action failing shapes: ")
                                for a in da:
                                    print(f"    layer shape: {a.shape}")
                                for a in batch_da:
                                    print(f"    batch layer shape: {a.shape}")
                                print(
                                    f"    batch sample shape: {batch_sample_da.shape}"  # type:ignore
                                )
                            if not ca_passing:
                                print(
                                    f"  Continuous action failing shape: {ca.shape}, batch shape {batch_ca.shape}"
                                )
                            if not lstd_passing:
                                print(
                                    f"  Continuous log probs not passing with shape {calp.shape}, batch shape {batch_calp.shape}"
                                )
    print(
        f"Stochastic Actor Continuous Actions   passed {ca_pass_count}/{total_tests} = {ca_pass_count/total_tests*100:.2f}%"
    )
    print(
        f"Stochastic Actor Continuous Log Probs passed {log_pass_count}/{total_tests} = {log_pass_count/total_tests*100:.2f}%"
    )
    print(
        f"Stochastic Actor Discrete Actions     passed {da_pass_count}/{total_tests} = {da_pass_count/total_tests*100:.2f}%"
    )


# %%
if __name__ == "__main__":
    # data = torch.from_numpy(np.array([[0.0, 1.1, -1.1, 2.0], [0.1, 1.2, -1.3, 2.4]]))
    # sa = StochasticActor(
    #     obs_dim=4,
    #     continuous_action_dim=4,
    #     max_actions=np.array([2, 2, 2, 3]),
    #     min_actions=np.array([-1, -1, -1, -3]),
    #     discrete_action_dims=[2, 3],
    #     hidden_dims=np.array([16, 32]),
    #     action_head_hidden_dims=np.array([48]),
    #     std_type="full",
    # )
    # print(
    #     sa(
    #         data[0],
    #     )
    # )
    QS_test()
    SA_test()

# %%
