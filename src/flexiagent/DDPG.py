import torch
import torch.nn.functional as F
import numpy as np
from Agent import Agent, MixedActor, ValueSA
from Util import T, get_multi_discrete_one_hot
from flexibuff import FlexiBatch
import os


class DDPG(Agent):
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
        name="Test_ddpg",
        device="cpu",
        eval_mode=False,
        gumbel_tau=0.5,
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

        self.total_action_dim = continuous_action_dim + np.sum(
            np.array(discrete_action_dims)
        )
        self.target_update_percentage = target_update_percentage
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
        )
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
        )
        self.discrete_action_dims = discrete_action_dims
        self.continuous_action_dim = continuous_action_dim
        self.action_noise = action_noise
        self.step = 0
        self.rl_step = 0
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.to(device)
        self.actor_target.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = ValueSA(
            obs_dim, self.total_action_dim, hidden_dim=256, device=device
        )
        self.critic_target = ValueSA(
            obs_dim, self.total_action_dim, hidden_dim=256, device=device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.to(device)
        self.critic_target.to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.device = device

    def __noise__(self, continuous_actions: torch.Tensor):
        noise = torch.normal(
            0,
            self.action_noise,
            (continuous_actions.shape[0], self.continuous_action_dim),
        ).to(self.device)
        if noise.shape[0] == 1:
            noise = noise.squeeze(0)
        return noise

    def train_actions(self, observations, action_mask=None, step=False, debug=False):
        observations = T(observations, self.device, debug=debug)
        if debug:
            print("DDPG train_actions Observations: ", observations)
        if step:
            self.step += 1
        with torch.no_grad():
            continuous_actions, discrete_action_activations = self.actor(
                x=observations, action_mask=action_mask, gumbel=True, debug=debug
            )

            continuous_logprobs = None
            discrete_logprobs = None

            if debug:
                print("DDPG train_actions continuous_actions: ", continuous_actions)
                print(
                    "DDPG train_actions discrete_action_activations: ",
                    discrete_action_activations,
                )
                print("DDPG noise: ", self.__noise__(continuous_actions))

            value = self.critic(
                x=observations,
                u=torch.cat(
                    (
                        continuous_actions + self.__noise__(continuous_actions),
                        discrete_action_activations[0],
                    ),  # TODO: Cat all discrete actions
                    dim=-1,
                ),
                debug=debug,
            )
            if len(observations.shape) > 1:
                discrete_actions = torch.zeros(
                    (observations.shape[0], len(discrete_action_activations)),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                discrete_actions = torch.zeros(
                    (1, len(discrete_action_activations)),
                    device=self.device,
                    dtype=torch.long,
                )
            if debug:
                print("DDPG discrete_action_activtions: ", discrete_action_activations)
            for i, activation in enumerate(discrete_action_activations):
                if debug:
                    print("DDPG train_actions activation: ", activation)
                discrete_actions[:, i] = torch.argmax(activation, dim=-1)

            if debug:
                print(
                    "DDPG train_actions discrete_actions after argmax: ",
                    discrete_actions,
                )

            discrete_actions = discrete_actions.detach().cpu().numpy()
            continuous_actions = continuous_actions.detach().cpu().numpy()
            return (
                discrete_actions,
                continuous_actions,
                discrete_logprobs,
                continuous_logprobs,
                value.detach().cpu().numpy(),
            )

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

            if len(discrete_action_activations_) == 1:
                daa_ = discrete_action_activations_[0]
            else:
                daa_ = torch.cat(discrete_action_activations_, dim=-1)

            if debug:
                print(
                    "DDPG reinforcement_learn continuous_actions_: ",
                    continuous_actions_,
                )
                print(
                    "DDPG reinforcement_learn discrete_action_activations_: ",
                    discrete_action_activations_,
                )
                print("DDPG reinforcement_learn daa: ", daa_)
                # input()
            actions_ = torch.cat([continuous_actions_, daa_], dim=-1)
            qtarget = self.critic_target(batch.obs_[agent_num], actions_).squeeze(-1)
            # TODO configure reward channel beyong just global_rewards
            next_q_value = (
                batch.global_rewards + (1 - batch.terminated) * self.gamma * qtarget
            )
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
        q_values = self.critic(batch.obs[agent_num], actions).squeeze(-1)
        qf1_loss = F.mse_loss(q_values, next_q_value)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        qf1_loss.backward()
        self.critic_optimizer.step()
        closs_item = qf1_loss.item()

        if self.rl_step % self.policy_frequency == 0 and not critic_only:
            c_act, d_act = self.actor(batch.obs[agent_num], mask)

            # TODO Check and make sure that the discrete actions are concatenated correctly
            if len(d_act) == 1:
                d_act = d_act[0]
            else:
                d_act = torch.cat(d_act, dim=-1)
            actor_loss = -self.critic(
                batch.obs[agent_num], torch.cat([c_act, d_act], dim=-1)
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.target_update_percentage * param.data
                    + (1 - self.target_update_percentage) * target_param.data
                )
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.target_update_percentage * param.data
                    + (1 - self.target_update_percentage) * target_param.data
                )
            aloss_item = actor_loss.item()
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
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.target_update_percentage * param.data
                + (1 - self.target_update_percentage) * target_param.data
            )
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.target_update_percentage * param.data
                + (1 - self.target_update_percentage) * target_param.data
            )
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
        torch.save(self.critic.state_dict(), checkpoint_path + "critic")
        torch.save(self.critic_target.state_dict(), checkpoint_path + "critic_target")
        torch.save(self.actor.state_dict(), checkpoint_path + "actor")
        torch.save(self.actor_target.state_dict(), checkpoint_path + "actor_target")

    def load(self, checkpoint_path):
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"
        self.actor.load_state_dict(torch.load(checkpoint_path + "actor"))
        self.actor_target.load_state_dict(torch.load(checkpoint_path + "actor_target"))
        self.critic.load_state_dict(torch.load(checkpoint_path + "critic"))
        self.critic_target.load_state_dict(
            torch.load(checkpoint_path + "critic_target")
        )
