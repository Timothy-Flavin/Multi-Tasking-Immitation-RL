from flexibuff import FlexibleBuffer
from DDPG import DDPG
import gymnasium as gym
import numpy as np


def test_single_env(
    env: gym.Env,
    agent: DDPG,
    buffer: FlexibleBuffer,
    n_episodes=100,
    joint_obs_dim=7,
    discrete=False,
    debug=False,
):
    agent: DDPG
    rewards = []
    for episode in range(n_episodes):
        ep_reward = 0
        obs, info = discrete_env.reset()
        obs = np.pad(obs, (0, joint_obs_dim - len(obs)), "constant")
        terminated, truncated = False, False
        while not (terminated or truncated):
            discrete_actions, continuous_actions, cont_lp, disc_lp, value = (
                agent.train_actions(obs, step=True, debug=debug)
            )
            if discrete:
                actions = discrete_actions[0, 0]
            else:
                actions = continuous_actions
            # print(
            #    f"c_a: {continuous_actions}, d_a: {discrete_actions}, actions: {actions}"
            # )

            obs_, reward, terminated, truncated, _ = discrete_env.step(actions)
            obs_ = np.pad(obs_, (0, joint_obs_dim - len(obs_)), "constant")

            buffer.save_transition(
                obs=obs,
                obs_=obs_,
                continuous_actions=continuous_actions,
                discrete_actions=discrete_actions,
                global_reward=reward,
                terminated=terminated,
            )
            # discrete_env.render()
            obs = obs_
            ep_reward += reward

        episodes = buffer.sample_episodes(256, as_torch=True)
        for ep in episodes:
            closs, aloss = agent.reinforcement_learn(ep, agent_num=0, debug=debug)
            # print(aloss, closs)
        print(f"n_ep: {episode} r: {ep_reward}")
        rewards.append(ep_reward)
    discrete_env.close()
    return rewards


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    discrete_env = gym.make("CartPole-v1")  # , render_mode="human")
    continuous_env = gym.make("Pendulum-v1")
    joint_obs_dim = (
        discrete_env.observation_space.shape[0]
        + continuous_env.observation_space.shape[0]
    )
    print("Making DDPG Model")
    models = [
        DDPG(
            obs_dim=joint_obs_dim,
            discrete_action_dims=[discrete_env.action_space.n],
            continuous_action_dim=continuous_env.action_space.shape[0],
            min_actions=continuous_env.action_space.low,
            max_actions=continuous_env.action_space.high,
            hidden_dims=np.array([128, 128]),
            gamma=0.99,
            policy_frequency=2,
            name="DDPG_cd_test",
            device="cuda",
            eval_mode=False,
        )
    ]
    print("Making Discrete Flexible Buffers")

    disc_mem_buffer = FlexibleBuffer(
        num_steps=50000,
        obs_size=joint_obs_dim,
        action_mask=None,
        discrete_action_cardinalities=[discrete_env.action_space.n],
        continuous_action_dimension=continuous_env.action_space.shape[0],
        path="./test_mem_buffer/",
        name="joint_buffer",
        n_agents=1,
        state_size=None,
        global_reward=True,
        log_prob_discrete=False,
    )
    print("Making Continuous Flexible Buffers")
    cont_mem_buffer = FlexibleBuffer(
        num_steps=50000,
        obs_size=joint_obs_dim,
        action_mask=None,
        discrete_action_cardinalities=[discrete_env.action_space.n],
        continuous_action_dimension=continuous_env.action_space.shape[0],
        path="./test_mem_buffer/",
        name="joint_buffer",
        n_agents=1,
        state_size=None,
        global_reward=True,
        log_prob_discrete=False,
    )
    print("Testing Discrete Environment")
    rewards = test_single_env(
        discrete_env, models[0], disc_mem_buffer, n_episodes=1000, discrete=True
    )
    print(rewards)
    plt.plot(rewards)
    plt.show()
