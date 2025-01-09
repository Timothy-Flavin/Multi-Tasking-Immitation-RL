from flexibuff import FlexibleBuffer
from DDPG import DDPG
from TD3 import TD3
from PPO import PPO
import gymnasium as gym
import numpy as np
from Agent import Agent
from typing import List
from PG import PG
import torch


gym_disc_env = "CartPole-v1"
gym_cont_env = "Pendulum-v1"  # "HalfCheetah-v4"


def test_single_env(
    env: gym.Env,
    agent: DDPG,
    buffer: FlexibleBuffer,
    n_episodes=100,
    joint_obs_dim=7,
    discrete=False,
    debug=False,
    online=False,
):
    agent: DDPG
    rewards = []
    aloss_return = []
    closs_return = []
    term_array = []
    step = 0
    for episode in range(n_episodes):
        m_aloss, m_closs = 0, 0
        ep_reward = 0
        obs, info = env.reset()
        obs = np.pad(obs, (0, joint_obs_dim - len(obs)), "constant")
        terminated, truncated = False, False
        while not (terminated or truncated):
            step += 1

            discrete_actions, continuous_actions, disc_lp, cont_lp, value = (
                agent.train_actions(obs, step=True, debug=debug)
            )
            # print(discrete_actions, continuous_actions)
            if discrete:
                actions = int(discrete_actions[0])
            else:
                actions = continuous_actions

            # print(actions)
            if cont_lp is None:
                cont_lp = 0
            if disc_lp is None:
                disc_lp = 0
            # print(actions)
            # print(
            #     f"c_a: {continuous_actions}, d_a: {discrete_actions}, actions: {actions}"
            # )

            obs_, reward, terminated, truncated, _ = env.step(actions)
            term_array.append(actions)
            obs_ = np.pad(obs_, (0, joint_obs_dim - len(obs_)), "constant")

            # print(disc_lp, cont_lp)
            buffer.save_transition(
                obs=obs,
                obs_=obs_,
                continuous_actions=continuous_actions,
                discrete_actions=discrete_actions,
                global_reward=reward,  # + abs(obs[1]) * 100,
                terminated=terminated or truncated,
                discrete_log_probs=disc_lp,
                continuous_log_probs=cont_lp,
            )
            # env.render()
            # print(abs(obs[1]) * 50)
            ep_reward += reward  # + abs(obs[1]) * 100
            obs = obs_

            # print(buffer.steps_recorded)
            if (
                buffer.steps_recorded > 255
                and buffer.episode_inds is not None
                and not online
            ) or (
                # and len(buffer.episode_inds) > 5
                online
                and buffer.steps_recorded > 2047
            ):
                # print(online)
                if online:
                    batch = buffer.sample_transitions(
                        idx=np.arange(0, buffer.steps_recorded), as_torch=True
                    )
                    # agent.mini_batch_size = buffer.steps_recorded - 1
                    # print(buffer.steps_recorded - 1)
                    # print(batch.discrete_actions[0, :, 0])
                    # print(torch.from_numpy(np.array(term_array)).to("cuda:0"))
                    # print(batch.discrete_log_probs[0, :, 0])
                    # input()
                    # print(batch.global_rewards)
                else:
                    batch = buffer.sample_transitions(batch_size=256, as_torch=True)
                # for ep in episodes:
                aloss, closs = agent.reinforcement_learn(
                    batch, agent_num=0, debug=debug
                )
                m_aloss += aloss
                m_closs += closs
                if online:
                    buffer.reset()
                    term_array = []
        # print(aloss, closs)
        aloss_return.append(m_aloss)
        closs_return.append(m_closs)
        rewards.append(ep_reward)
        er = 10
        if episode % er == 0 and episode > 1:
            print(
                f"n_ep: {episode} r: {sum(rewards[-10:])/10}, step: {step}, best: {max(rewards[-10:])} m_aloss: {m_aloss}, m_closs: {m_closs}"
            )
            # print(f"lr: {agent.optimizer.param_groups[0]["lr"]}, G: {agent.g_mean}")
            # plt.plot(rewards)
            # plt.show()
            if episode % er == 0:
                print("human animating")
                env = gym.make(
                    gym_disc_env if discrete else gym_cont_env, render_mode="human"
                )
        if episode % er == 1 and episode > 1:
            print("no more human")
            env.close()
            env = gym.make(id=gym_disc_env if discrete else gym_cont_env)
        # env = gym.make("CartPole-v1")

    env.close()
    return rewards, aloss_return, closs_return


def test_dual_env(
    discrete_env: gym.Env,
    continuous_env: gym.Env,
    agent: DDPG,
    buffer: FlexibleBuffer,
    n_steps=50000,
    joint_obs_dim=7,
    debug=False,
    online=False,
):
    agent: DDPG
    rewards = [[], []]
    step = 0
    ep_rewards = [0, 0]
    episode_nums = [0, 0]
    d_obs, d_info = discrete_env.reset()
    c_obs, c_info = continuous_env.reset()
    obs = np.concatenate([d_obs, c_obs])

    while step < n_steps:
        discrete_actions, continuous_actions, cont_lp, disc_lp, value = (
            agent.train_actions(obs, step=True, debug=debug)
        )
        if disc_lp is None:
            disc_lp = 0
        if cont_lp is None:
            cont_lp = 0

        discrete_actions = discrete_actions[0]
        continuous_actions = continuous_actions
        # print(f"disc: {discrete_actions}, cont: {continuous_actions}")
        d_obs_, d_reward, d_terminated, d_truncated, _ = discrete_env.step(
            discrete_actions
        )
        c_obs_, c_reward, c_terminated, c_truncated, _ = continuous_env.step(
            continuous_actions
        )

        obs_ = np.concatenate([d_obs_, c_obs_])

        buffer.save_transition(
            obs=obs,
            obs_=obs_,
            continuous_actions=continuous_actions,
            discrete_actions=discrete_actions,
            global_reward=d_reward + (c_reward + 1000) / 30,
            terminated=d_terminated,
            discrete_log_probs=disc_lp,
            continuous_log_probs=cont_lp,
        )
        # env.render()
        ep_rewards[0] += d_reward
        ep_rewards[1] += c_reward

        if d_terminated or d_truncated:
            print(
                f"n_ep: {episode_nums} r: {ep_rewards[0]},{(ep_rewards[1]+1000)/30}, step: {step}"
            )
            d_obs_, d_info = discrete_env.reset()
            rewards[0].append(ep_rewards[0])
            ep_rewards[0] = 0
            episode_nums[0] += 1
        if c_terminated or c_truncated:
            print(
                f"n_ep: {episode_nums} r: {ep_rewards[0]},{(ep_rewards[1]+1000)/30}, step: {step}"
            )
            c_obs_, c_info = continuous_env.reset()
            rewards[1].append(ep_rewards[1])
            ep_rewards[1] = 0
            episode_nums[1] += 1

        obs = np.concatenate([d_obs_, c_obs_])
        step += 1
        if buffer.steps_recorded > 256 and buffer.episode_inds is not None:
            episodes = buffer.sample_episodes(256, as_torch=True)
            for ep in episodes:
                closs, aloss = agent.reinforcement_learn(ep, agent_num=0, debug=debug)
            if online:
                buffer.reset()
        # print(aloss, closs)

    return rewards


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    discrete_env = gym.make(
        gym_disc_env
    )  # """MountainCar-v0")  # )   # , render_mode="human")
    continuous_env = gym.make(gym_cont_env)
    joint_obs_dim = (
        discrete_env.observation_space.shape[0]
        + continuous_env.observation_space.shape[0]
    )

    def make_models():
        print("Making Model")
        names = ["PG", "TD3", "DDPG"]
        print(
            continuous_env.action_space.low,
            continuous_env.action_space.high,
            continuous_env.action_space.shape,
            continuous_env.action_space.shape[0],
        )
        models = [
            PG(
                obs_dim=joint_obs_dim,
                discrete_action_dims=[discrete_env.action_space.n],
                continuous_action_dim=continuous_env.action_space.shape[0],
                hidden_dims=np.array([64, 64]),
                min_actions=continuous_env.action_space.low,
                max_actions=continuous_env.action_space.high,
                gamma=0.99,
                device="cuda",
                entropy_loss=0,
                mini_batch_size=64,
                n_epochs=5,
                lr=0.1,
                advantage_type="constant",
                norm_advantages=False,
                anneal_lr=2000000,
                value_loss_coef=0.1,  # 5,
                ppo_clip=0.1,
                value_clip=0.5,
                orthogonal=True,
                activation="tanh",
                starting_actorlogstd=0,
            ),
            TD3(
                obs_dim=joint_obs_dim,
                discrete_action_dims=[discrete_env.action_space.n],
                continuous_action_dim=continuous_env.action_space.shape[0],
                min_actions=continuous_env.action_space.low,
                max_actions=continuous_env.action_space.high,
                hidden_dims=np.array([64, 64]),
                gamma=0.99,
                policy_frequency=2,
                name="TD3_cd_test",
                device="cuda",
                eval_mode=False,
                rand_steps=10000,
            ),
            DDPG(
                obs_dim=joint_obs_dim,
                discrete_action_dims=[discrete_env.action_space.n],
                continuous_action_dim=continuous_env.action_space.shape[0],
                min_actions=continuous_env.action_space.low,
                max_actions=continuous_env.action_space.high,
                hidden_dims=np.array([256, 256]),
                gamma=0.99,
                policy_frequency=4,
                name="TD3_cd_test",
                device="cuda",
                eval_mode=False,
                rand_steps=10000,
            ),
            # PPO(
            #     obs_dim=joint_obs_dim,
            #     discrete_action_dims=[discrete_env.action_space.n],
            #     continuous_action_dim=continuous_env.action_space.shape[0],
            #     hidden_dims=np.array([128, 128]),
            #     min_actions=continuous_env.action_space.low,
            #     max_actions=continuous_env.action_space.high,
            #     gamma=0.99,
            #     device="cuda",
            #     entropy_loss=0.05,
            # ),
        ]
        return models, names

    print("Making Discrete Flexible Buffers")

    mem_buffer = FlexibleBuffer(
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
        log_prob_discrete=True,
        log_prob_continuous=continuous_env.action_space.shape[0],
    )

    models, names = make_models()

    results = {}
    for n in names:
        results[n] = []

    for n in range(len(names)):
        print("Testing Model ", names[n])

        mem_buffer.reset()
        print("Testing Continuous Environment")
        rewards, aloss, closs = test_single_env(
            continuous_env,
            agent=models[n],
            buffer=mem_buffer,
            n_episodes=2000 if names[n] == "PG" else 200,
            discrete=False,
            joint_obs_dim=joint_obs_dim,
            online=names[n] in ["PPO", "PG"],
        )
        print(rewards)
        plt.plot(rewards)
        plt.show()

        plt.plot(aloss)
        plt.plot(closs)
        plt.legend(["actor", "critic"])
        plt.show()
        models[n].save(f"../../TestModels/{names[n]}_Continuous")

        models, names = make_models()
        mem_buffer.reset()
        print("Testing Discrete Environment")
        rewards, aloss, closs = test_single_env(
            env=discrete_env,
            agent=models[n],
            buffer=mem_buffer,
            n_episodes=2000 if names[n] == "PG" else 1000,
            discrete=True,
            joint_obs_dim=joint_obs_dim,
            online=names[n] in ["PPO", "PG"],
        )
        print(rewards)
        plt.plot(rewards)
        plt.show()
        am = np.abs(aloss).max()
        cm = np.abs(closs).max()
        plt.plot(aloss / am)
        plt.plot(closs / cm)
        plt.legend([f"actor {am}", f"critic {cm}"])
        plt.show()
        models[n].save(f"../../TestModels/{names[n]}_Discrete")

        continue

        models, names = make_models()
        mem_buffer.reset()
        print("Testing Dual Environment")
        r1, r2 = test_dual_env(
            discrete_env=discrete_env,
            continuous_env=continuous_env,
            agent=models[0],
            buffer=mem_buffer,
            n_steps=25000,
            joint_obs_dim=joint_obs_dim,
            debug=False,
        )
        plt.plot(aloss)
        plt.plot(closs)
        plt.legend(["actor", "critic"])
        plt.show()
        models[n].save(f"../../TestModels/{names[n]}_Dual")

    r1 = np.array(r1)
    r2 = np.array(r2)
    print(r1)
    print(r2)
    plt.plot(r1 / np.abs(r1).max())
    plt.plot(r2 / np.abs(r2).max())
    plt.title(f"m1: {r1.max()}, m2: {r2.max()}")
    plt.show()
