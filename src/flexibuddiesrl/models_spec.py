from flexibuddiesrl.DQN import DQN
from flexibuddiesrl.PG import PG
from flexibuddiesrl.DDPG import DDPG
from flexibuddiesrl.TD3 import TD3
from flexibuddiesrl.Agent import QS
from flexibuddiesrl.Agent import Agent

from flexibuff import FlexibleBuffer, FlexiBatch
import matplotlib.pyplot as plt
import numpy as np


def test_imitation_learn(agent: Agent, batch: FlexiBatch):
    dlosses, alosses = [], []
    for i in range(10):
        dloss, closs = agent.imitation_learn(
            batch.obs[0], batch.continuous_actions[0], batch.discrete_actions[0]
        )
        dlosses.append(dloss)
        alosses.append(closs)
    return dlosses, alosses


def set_up_memory_buffer(
    obs_dim, continuous_action_dims, discrete_action_dims, termination
):
    obs_batch = np.random.rand(14, obs_dim).astype(np.float32)
    obs_batch_ = obs_batch + 0.1
    dacs = np.zeros((14, len(discrete_action_dims)), dtype=np.int64)
    for i, dim in enumerate(discrete_action_dims):
        dacs[:, i] = np.random.randint(0, dim, size=14)

    # Set up memory buffer
    mem = FlexibleBuffer(
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
        mem.save_transition(
            terminated=termination[i],
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
    return mem


def dqn_agents(obs_dim, continuous_action_dim, discrete_action_dims):
    duel_tests = [True, False]
    dis_tests = [None, discrete_action_dims]
    con_tests = [0, continuous_action_dim]
    head_hidden_tests = [None, 64]
    epsilon_tests = [1.0, 0.0]
    conservative_tests = [True, False]
    entropy_tests = [0.0, 0.1]
    munchausen_tests = [0.0, 0.9]
    agents = []
    agent_parameters = []
    for duel in duel_tests:
        for dis in dis_tests:
            for con in con_tests:
                for head_hidden in head_hidden_tests:
                    for eps in epsilon_tests:
                        for cql in conservative_tests:
                            for ent in entropy_tests:
                                for mun in munchausen_tests:
                                    if mun > 0 and ent <= 0:
                                        continue
                                    agent_parameters.append(
                                        {
                                            "duel": duel,
                                            "discrete_action_dims": dis,
                                            "continuous_action_dim": con,
                                            "head_hidden": head_hidden,
                                            "epsilon": eps,
                                            "cql": cql,
                                            "entropy": ent,
                                            "munchausen": mun,
                                        }
                                    )
                                    agent = DQN(
                                        obs_dim=obs_dim,
                                        continuous_action_dims=con,
                                        max_actions=np.array([1, 2]),
                                        min_actions=np.array([0, 0]),
                                        discrete_action_dims=dis,
                                        hidden_dims=[32, 32],
                                        device="cuda:0",
                                        lr=0.001,
                                        imitation_lr=0.001,
                                        activation="relu",
                                        dueling=duel,
                                        n_c_action_bins=5,
                                        head_hidden_dim=head_hidden,
                                        conservative=cql,
                                        init_eps=eps,
                                    )
                                    agents.append(agent)

    print(f"Total DQN agents created: {len(agents)}")
    return agents, agent_parameters


def pg_agents(obs_dim, continuous_action_dim, discrete_action_dims):
    agents = []
    agent_parameters = []

    for dis in [None, discrete_action_dims]:
        for cdim in [0, continuous_action_dim]:
            for ent in [0.0, 0.1]:
                for ppc in [0.0, 0.2]:
                    for vclip in [0.0, 0.5]:
                        for advt in ["gae", "gv", "a2c"]:
                            for cg in [True, False]:
                                if dis is None and cdim == 0:
                                    print(
                                        "Skipping agent with no discrete or continuous actions"
                                    )
                                    continue
                                agent_parameters.append(
                                    {
                                        "discrete_action_dims": dis,
                                        "continuous_action_dim": continuous_action_dim,
                                        "entropy_regularization": ent,
                                        "ppo_clip": ppc,
                                        "value_clip": vclip,
                                        "advantage_type": advt,
                                        "clip_grad": cg,
                                    }
                                )
                                agent = PG(
                                    obs_dim=obs_dim,
                                    continuous_action_dim=cdim,
                                    discrete_action_dims=dis,
                                    max_actions=np.array([1, 2]),
                                    min_actions=np.array([0, 0]),
                                    lr=0.001,
                                    gamma=0.99,
                                    n_epochs=4,
                                    device="cuda:0",
                                    entropy_loss=ent,
                                    ppo_clip=ppc,
                                    value_clip=vclip,
                                    advantage_type=advt,
                                    mini_batch_size=4,
                                    clip_grad=cg,
                                )
                                agents.append(agent)

    print(f"Total PG agents created: {len(agents)}")
    return agents, agent_parameters


if __name__ == "__main__":
    # Deciding the dimensions to be used for the test
    obs_dim = 3
    continuous_action_dim = 2
    discrete_action_dims = [4, 5, 6]
    termination = np.zeros(14, dtype=np.float32)
    termination[4] = 1.0  # Setting terminations for deterministic testing
    termination[10] = 1.0

    testable_model_functions = {"DQN": dqn_agents, "PG": pg_agents}
    algorithm = input(
        f"Which model should be tested?: {testable_model_functions.keys()} \n"
    ).upper()
    while algorithm not in testable_model_functions:
        algorithm = input(
            f"Please enter a valid model name ({testable_model_functions.keys()}): "
        )

    mem = set_up_memory_buffer(
        obs_dim, continuous_action_dim, discrete_action_dims, termination
    )

    print(mem)
    # setting up single and multiple observations and actions
    # to test train_action, immitiation_learn, and reinforcement_learn methods
    obs = np.random.rand(obs_dim).astype(np.float32)
    obs_ = np.random.rand(obs_dim).astype(np.float32)

    agents, agent_params = testable_model_functions[algorithm](
        obs_dim, continuous_action_dim, discrete_action_dims
    )
    x = ""
    for i in range(len(agents)):
        print(f"Agent {i}: {agent_params[i]}")
        d_acts, c_acts, d_log, c_log, _ = agents[i].train_actions(
            obs, step=True, debug=True
        )
        print(
            f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}"
        )
        aloss, closs = agents[i].reinforcement_learn(
            mem.sample_transitions(12, as_torch=True), 0, critic_only=False, debug=False
        )
        print(f"Reinforcement learn losses: aloss: {aloss}, closs: {closs}")

        aloss, closs = test_imitation_learn(
            agents[i],
            mem.sample_transitions(14, as_torch=True, device="cuda:0"),
        )
        print(f"Imitation learn losses: aloss: {aloss}, closs: {closs}")
        if x.lower() == "auto":
            continue
        x = input(
            f"Press enter to continue to the next agent, or type 'exit' to quit, or 'auto' to skip inputs: "
        )
        if x.lower() == "exit":
            break
