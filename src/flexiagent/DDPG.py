import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from Agent import Agent, MixedActor, ValueSA
from Util import T
from flexibuff import Flexibatch


class DDPG(Agent):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[],
        max_actions=[],
        min_actions=[],
        action_noise=0.1,
        device="cpu",
    ):
        assert not (
            continuous_action_dim == None and discrete_action_dims == None
        ), "At least one action dim should be provided"
        assert len(max_actions) == len(discrete_action_dims) and len(
            min_actions
        ) == len(
            discrete_action_dims
        ), "max_actions should be provided for each discrete action dim"

        self.total_action_dim = continuous_action_dim + np.sum(
            np.array(discrete_action_dims)
        )

        self.actor = MixedActor(
            obs_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=np.array([1, 1]),
            min_actions=np.array([-1, -1]),
            device=device,
            hidden_dims=np.array([256, 256]),
            encoder=None,
            device=device,
            tau=0.5,
            hard=False,
        )
        self.actor_target = MixedActor(
            obs_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=np.array([1, 1]),
            min_actions=np.array([-1, -1]),
            device=device,
            hidden_dims=np.array([256, 256]),
            encoder=None,
            device=device,
            tau=0.5,
            hard=False,
        )
        self.discrete_action_dims = discrete_action_dims
        self.continuous_action_dim = continuous_action_dim
        self.action_noise = action_noise
        self.step = 0
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
        return torch.normal(
            0,
            self.action_noise,
            (continuous_actions.shape[0], self.continuous_action_dim),
        ).to(self.device)

    def train_actions(self, observations, legal_actions=None, step=False):
        observations = T(observations, self.device)
        if step:
            self.step += 1

        continuous_actions, discrete_action_activations = self.actor(
            observations, legal_actions
        )

        continuous_logprobs = None
        discrete_logprobs = None

        value = self.critic(
            observations,
            continuous_actions + self.__noise__(continuous_actions),
            discrete_action_activations,
        )

        discrete_actions = torch.zeros(
            (observations.shape[0], len(discrete_action_activations)),
            device=self.device,
            dtype=torch.long,
        )
        for i, activation in enumerate(discrete_action_activations):
            discrete_actions[:, i] = torch.argmax(activation, dim=1)

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
        self, batch: Flexibatch, agent_num=0, critic_only=False, debug=False
    ):

        return 0
