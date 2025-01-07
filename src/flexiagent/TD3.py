import torch
import torch.nn.functional as F
import numpy as np
from Agent import Agent, MixedActor, ValueSA
from Util import T, get_multi_discrete_one_hot
from flexibuff import FlexiBatch
import os


class TD3(Agent):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[],
        max_actions=[],
        min_actions=[],
        action_noise=0.1,
        hidden_dims=np.array([256, 256]),
        gamma=0.99,
        policy_frequency=2,
        target_update_percentage=0.01,
        name="Test_TD3",
        device="cpu",
        eval_mode=False,
        gumbel_tau=0.25,
        rand_steps=10000,
    ):
        # documentation
        """
        obs_dim: int
            The dimension of the observation space
        continuous_action_dim: int
        discrete_action_dims: list
            The cardonality of each discrete action space
        max_actions: list
            The maximum value of the continuous action space
        min_actions: list
            The minimum value of the continuous action space
        action_noise: float
            The noise to add to the policy output into the value function
        hidden_dims: list
            The hidden dimensions of the actor and critic
        gamma: float
            The discount factor
        policy_frequency: int
            The frequency of policy updates
        target_update_percentage: float
            The percentage of the target network to update
        name: str
            The name of the agent
        device: str
        """
        assert not (
            continuous_action_dim is None and discrete_action_dims is None
        ), "At least one action dim should be provided"
        assert len(max_actions) == len(discrete_action_dims) and len(
            min_actions
        ) == len(
            discrete_action_dims
        ), "max_actions should be provided for each discrete action dim"

        self.device = device
        self.total_action_dim = continuous_action_dim + np.sum(
            np.array(discrete_action_dims)
        )
        self.target_update_percentage = target_update_percentage
        self.rand_steps = rand_steps
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.eval_mode = eval_mode
        self.name = name
        self.actor = MixedActor(
            obs_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            device=device,
            hidden_dims=hidden_dims,
            encoder=None,
            tau=gumbel_tau,
            hard=False,
        ).float()
        self.actor_target = MixedActor(
            obs_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            device=device,
            hidden_dims=hidden_dims,
            encoder=None,
            tau=0.3,
            hard=False,
        ).float()
        if continuous_action_dim > 0:
            self.min_actions = torch.from_numpy(np.array(min_actions)).to(self.device)
            self.max_actions = torch.from_numpy(np.array(max_actions)).to(self.device)

        self.discrete_action_dims = discrete_action_dims
        self.continuous_action_dim = continuous_action_dim
        self.action_noise = action_noise
        self.step = 0
        self.rl_step = 0
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.to(device)
        self.actor_target.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic1 = ValueSA(
            obs_dim, self.total_action_dim, hidden_dim=256, device=device
        ).float()
        self.critic2 = ValueSA(
            obs_dim, self.total_action_dim, hidden_dim=256, device=device
        ).float()
        self.critic1_target = ValueSA(
            obs_dim, self.total_action_dim, hidden_dim=256, device=device
        ).float()
        self.critic2_target = ValueSA(
            obs_dim, self.total_action_dim, hidden_dim=256, device=device
        ).float()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        # self.critic2.load_state_dict(self.critic1.state_dict())
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters())
        )

    def __noise__(self, continuous_actions: torch.Tensor):
        noise = torch.normal(
            0,
            self.action_noise,
            continuous_actions.shape,
        ).to(self.device)
        # if noise.shape[0] == 1:
        # noise = noise.squeeze(0)
        return noise

    def _add_noise(self, continuous_actions):
        if self.continuous_action_dim == 0:
            return 0
        return torch.clip(
            continuous_actions + self.__noise__(continuous_actions),
            self.min_actions,
            self.max_actions,
        )

    def _get_random_actions(self, action_mask=None, debug=False):
        continuous_actions = (
            torch.rand(size=(self.continuous_action_dim,), device=self.device) * 2 - 1
        ) * self.actor.action_scales - self.actor.action_biases
        discrete_actions = torch.zeros(
            (len(self.discrete_action_dims),), device=self.device, dtype=torch.long
        )

        for dim, dim_size in enumerate(self.discrete_action_dims):
            discrete_actions[dim] = torch.randint(dim_size, (1,))
        return discrete_actions, continuous_actions

    def train_actions(self, observations, action_mask=None, step=False, debug=False):
        observations = T(observations, self.device, debug=debug)
        if debug:
            print("    TD3 train_actions Observations: ", observations)
        if step:
            self.step += 1
        if self.step < self.rand_steps:
            discrete_actions, continuous_actions = self._get_random_actions(
                action_mask, debug=debug
            )
            return (
                discrete_actions.detach().cpu().numpy(),
                continuous_actions.detach().cpu().numpy(),
                None,
                None,
                None,
            )
        with torch.no_grad():
            continuous_actions, discrete_action_activations = self.actor(
                x=observations, action_mask=action_mask, gumbel=True, debug=debug
            )
            continuous_actions_noisy = self._add_noise(continuous_actions)

            continuous_logprobs = None
            discrete_logprobs = None

            if debug:
                print("    TD3 train_actions continuous_actions: ", continuous_actions)
                print(
                    "    TD3 train_actions continuous_actions_noisy: ",
                    continuous_actions_noisy,
                )
                print(
                    "    TD3 train_actions discrete_action_activations: ",
                    discrete_action_activations,
                )
                print("    TD3 noise: ", self.__noise__(continuous_actions))
            # u = torch.cat(
            #     (
            #         continuous_actions_noisy,
            #         discrete_action_activations[0],
            #     ),  # TODO: Cat all discrete actions
            #     dim=-1,
            # )
            # value = self.critic1(
            #     x=observations,
            #     u=u,
            #     debug=debug,
            # )
            if len(observations.shape) > 1:
                discrete_actions = torch.zeros(
                    (observations.shape[0], len(discrete_action_activations)),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                discrete_actions = torch.zeros(
                    (len(discrete_action_activations)),  # was (1,len...)
                    device=self.device,
                    dtype=torch.long,
                )
            if debug:
                print(
                    "    TD3 discrete_action_activtions: ", discrete_action_activations
                )
            for i, activation in enumerate(discrete_action_activations):
                if debug:
                    print("    TD3 train_actions activation: ", activation)
                discrete_actions[i] = torch.argmax(activation, dim=-1)

            if debug:
                print(
                    "    TD3 train_actions discrete_actions after argmax: ",
                    discrete_actions,
                )

            discrete_actions = discrete_actions.detach().cpu().numpy()
            continuous_actions_noisy = continuous_actions_noisy.detach().cpu().numpy()
            return (
                discrete_actions,
                continuous_actions_noisy,
                discrete_logprobs,
                continuous_logprobs,
                0,  # value.detach().cpu().numpy(),
            )

    def polyak_update(self, tau=0.01):
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(
            self.critic1.parameters(), self.critic1_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(
            self.critic2.parameters(), self.critic2_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def reinforcement_learn(
        self, batch: FlexiBatch, agent_num=0, critic_only=False, debug=False
    ):
        aloss_item = 0
        closs_item = 0
        self.rl_step += 1
        with torch.no_grad():
            if batch.action_mask is not None:
                mask = batch.action_mask[agent_num]
                mask_ = batch.action_mask_[agent_num]
            else:
                mask = 1.0
                mask_ = 1.0
            continuous_actions_, discrete_action_activations_ = self.actor_target(
                batch.obs_[agent_num], mask_, gumbel=True
            )
            daa_ = discrete_action_activations_
            # if len(discrete_action_activations_) == 1:
            #     daa_ = discrete_action_activations_[0]
            # else:
            #     daa_ = torch.cat(discrete_action_activations_, dim=-1)

            if debug:
                print(
                    "TD3 reinforcement_learn continuous_actions_: ",
                    continuous_actions_,
                )
                print(
                    "TD3 reinforcement_learn discrete_action_activations_: ",
                    discrete_action_activations_,
                )
                print("TD3 reinforcement_learn daa_: ", daa_)
                # input()
            u_ = torch.cat([self._add_noise(continuous_actions_)] + daa_, dim=-1)

            if debug:
                print("u_: ", u_, "shape: ", u_.shape)
            qtarget = torch.minimum(
                self.critic1_target(x=batch.obs_[agent_num], u=u_),
                self.critic2_target(x=batch.obs_[agent_num], u=u_),
            ).squeeze(-1)
            if debug:
                print("TD3 reinforcement_learn qtarget: ", qtarget)
            # TODO configure reward channel beyong just global_rewards
            next_q_value = (
                batch.global_rewards + (1 - batch.terminated) * self.gamma * qtarget
            )
            if debug:
                print("TD3 reinforcement_learn next_q_value: ", next_q_value)

        # for each discrete action, get the one hot coding and concatinate them

        actions = torch.cat(
            [
                batch.continuous_actions[agent_num],
                get_multi_discrete_one_hot(
                    batch.discrete_actions[agent_num],
                    discrete_action_dims=self.discrete_action_dims,
                    debug=debug,
                ),
            ],
            dim=-1,
        )
        q1_values = self.critic1(batch.obs[agent_num], actions).squeeze(-1)
        q2_values = self.critic2(batch.obs[agent_num], actions).squeeze(-1)
        qf1_loss = F.mse_loss(q1_values, next_q_value)
        qf2_loss = F.mse_loss(q2_values, next_q_value)
        L = qf1_loss + qf2_loss

        # optimize the critic
        self.critic_optimizer.zero_grad()
        L.backward()
        self.critic_optimizer.step()

        if self.rl_step % self.policy_frequency == 0 and not critic_only:
            # print("   TD3 reinforcement_learn actor update")
            c_act, d_act = self.actor(batch.obs[agent_num], mask)

            # TODO Check and make sure that the discrete actions are concatenated correctly
            if len(d_act) == 1:
                d_act = d_act[0]
            else:
                d_act = torch.cat(d_act, dim=-1)
            actor_loss = -self.critic1(
                batch.obs[agent_num], torch.cat([c_act, d_act], dim=-1)
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            self.polyak_update(self.target_update_percentage)
            aloss_item = actor_loss.item()

        closs_item = L.item()
        return aloss_item, closs_item

    def ego_actions(self, observations, action_mask=None):
        with torch.no_grad():
            continuous_actions, discrete_action_activations = self.actor(
                observations, action_mask, gumbel=False
            )
            discrete_actions = torch.zeros(
                (observations.shape[0], len(discrete_action_activations)),
                device=self.device,
                dtype=torch.float32,
            )
            for i, activation in enumerate(discrete_action_activations):
                discrete_actions[:, i] = torch.argmax(activation, dim=1)
            return discrete_actions, continuous_actions

    def imitation_learn(self, observations, continuous_actions, discrete_actions):
        con_a, disc_a = self.actor.forward(observations, gumbel=False)
        loss = F.mse_loss(con_a, continuous_actions) + F.cross_entropy(
            disc_a, discrete_actions
        )
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # update the target network
        self.polyak_update(self.target_update_percentage)
        return loss

    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action):
        print("expected_V not implemeted")
        return 0

    def save(self, checkpoint_path):
        if self.eval_mode:
            print("Not saving because model in eval mode")
            return
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(self.critic1.state_dict(), checkpoint_path + "/critic1")
        torch.save(self.critic2.state_dict(), checkpoint_path + "/critic2")
        torch.save(
            self.critic1_target.state_dict(), checkpoint_path + "/critic1_target"
        )
        torch.save(
            self.critic2_target.state_dict(), checkpoint_path + "/critic2_target"
        )
        torch.save(self.actor.state_dict(), checkpoint_path + "/actor")
        torch.save(self.actor_target.state_dict(), checkpoint_path + "/actor_target")

    def load(self, checkpoint_path):
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"
        self.actor.load_state_dict(torch.load(checkpoint_path + "/actor"))
        self.actor_target.load_state_dict(torch.load(checkpoint_path + "/actor_target"))
        self.critic1.load_state_dict(torch.load(checkpoint_path + "/critic1"))
        self.critic2.load_state_dict(torch.load(checkpoint_path + "/critic2"))
        self.critic1_target.load_state_dict(
            torch.load(checkpoint_path + "/critic1_target")
        )
        self.critic2_target.load_state_dict(
            torch.load(checkpoint_path + "/critic2_target")
        )


if __name__ == "__main__":

    print("Testing TD3 functionality")

    c_act_dim = 2
    d_act_dims = [4, 3]
    obs = np.random.rand(10).astype(np.float32)
    obs_ = np.random.rand(10).astype(np.float32)
    obs_batch = np.random.rand(5, 10).astype(np.float32)
    obs_batch_ = obs_batch + 0.1

    dacs = np.stack(
        (np.random.randint(0, 4, size=(5)), np.random.randint(0, 3, size=(5))), axis=-1
    )
    dacs = np.array([dacs])
    print(dacs.shape)
    print(dacs)
    mem = FlexiBatch(
        obs=np.array([obs_batch]),
        obs_=np.array([obs_batch_]),
        continuous_actions=np.array([np.random.rand(5, 2).astype(np.float32)]),
        discrete_actions=dacs,
        global_rewards=np.random.rand(5).astype(np.float32),
        terminated=np.random.randint(0, 2, size=5),
    )
    mem.to_torch("cuda")

    td3 = TD3(
        obs_dim=10,
        continuous_action_dim=c_act_dim,
        discrete_action_dims=d_act_dims,
        max_actions=np.array([1, 1]),
        min_actions=np.array([-1, -1]),
        hidden_dims=[128, 128],
        gamma=0.99,
        policy_frequency=1,
        action_noise=0.05,
        name="TD3_Test",
        device="cuda",
        eval_mode=False,
        gumbel_tau=0.5,
        rand_steps=2,
    )

    # discrete_actions,
    # continuous_actions,
    # discrete_logprobs,
    # continuous_logprobs,
    # value
    print("Testing train_actions")
    print("act rand: ", td3.train_actions(obs, step=True, debug=True))
    print("act not_rand: ", td3.train_actions(obs, step=True, debug=True))
    print("act not_rand: ", td3.train_actions(obs, step=True, debug=True))

    print("Testing reinforcement_learn")
    print("rl: ", td3.reinforcement_learn(mem, agent_num=0, debug=True))
