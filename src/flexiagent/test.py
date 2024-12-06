from flexibuff import FlexiBatch, FlexibleBuffer
from DDPG import DDPG
import gymnasium as gym
import numpy as np

discrete_env = gym.make("CartPole-v1")
continuous_env = gym.make("Pendulum-v1")
Agent = DDPG(
    obs_dim=discrete_env.observation_space.shape[0]
    + continuous_env.observation_space.shape[0],
    discrete_action_dims=[discrete_env.action_space.shape[0]],
    continuous_action_dim=continuous_env.action_space.shape[0],
    min_actions=[discrete_env.action_space.low[0]],
    max_actions=[discrete_env.action_space.high[0]],
    hidden_dims=np.array([128, 128]),
    gamma=0.99,
    policy_frequency=2,
    name="DDPG_cp_pen_test",
    device="cuda",
    eval_mode=False,
)


def test_discrete_env(discrete_env, agent, buffer: FlexibleBuffer, n_episodes=100):
    for episode in range(n_episodes):
        obs = discrete_env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _ = discrete_env.step(action)
            discrete_env.render()
        discrete_env.close()
