import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from Agent import QS


class DQN(nn.Module):
    def __init__(
        self,
        obs_dim,
        discrete_action_dims=None,  # np.array([2]),
        continuous_action_dims=None,  # 2,
        min_actions=None,  # np.array([-1,-1]),
        max_actions=None,  # ,np.array([1,1]),
        hidden_dims=[64, 64],
        gamma=0.99,
        lr=3e-4,
        dueling=False,
        n_c_action_bins=10,
        munchausen=0,
        entropy=0,
        twin=False,
        delayed=False,
        activation="relu",
        orthogonal=False,
        action_epsilon=0.9,
        eps_decay_half_life=10000,
    ):
        super(DQN, self).__init__()
        self.obs_dim = obs_dim  # size of observation
        self.discrete_action_dims = (
            discrete_action_dims  # cardonality for each discrete action
        )
        self.continuous_action_dims = (
            continuous_action_dims  # number of continuous actions
        )
        self.min_actions = min_actions  # min continuous action value
        self.max_actions = max_actions  # max continuous action value
        if max_actions is not None:
            self.action_ranges = self.max_actions - self.min_actions
            self.action_means = (self.max_actions + self.min_actions) / 2
        self.gamma = gamma
        self.lr = lr
        self.dueling = (
            dueling  # whether or not to learn True: V+Adv = Q or False: Adv = Q
        )
        self.n_c_action_bins = n_c_action_bins  # number of discrete action bins to discretize continuous actions
        self.munchausen = munchausen  # use munchausen loss or not
        self.entropy_loss_coef = entropy  # use soft Q learning entropy loss or not H(Q)
        self.twin = False  # min(double q) to reduce bias
        self.init_eps = action_epsilon  # starting eps_greedy epsilon
        self.eps = self.init_eps
        self.half_life = eps_decay_half_life  # eps cut in half every 'half_life' frames
        self.step = 0
        self.Q1 = QS(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dims,
            discrete_action_dims=discrete_action_dims,
            hidden_dims=hidden_dims,
            activation=activation,
            orthogonal=orthogonal,
            dueling=dueling,
            n_c_action_bins=n_c_action_bins,
        )

        self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr)

    def train_actions(self, observations, action_mask=None, step=False):
        if self.init_eps > 0.0:
            self.eps = self.init_eps * self.step / (self.step + self.half_life)
        value = 0

        if self.init_eps > 0.0 and np.random.rand() < self.eps:
            if len(self.discrete_action_dims) > 0:
                disc_act = np.zeros(
                    shape=len(self.discrete_action_dims), dtype=np.int32
                )
                for i in range(len(self.discrete_action_dims)):
                    disc_act[i] = np.random.randint(0, self.discrete_action_dims[i])

            if len(self.continuous_action_dims) > 0:
                cont_act = np.zeros(
                    shape=len(self.continuous_action_dims), dtype=np.int32
                )
                for i in range(len(self.continuous_action_dims)):
                    cont_act[i] = np.random.randint(
                        0, self.n_c_action_bins, dtype=np.float32
                    )
                cont_act = (
                    cont_act / (self.n_c_action_bins - 1)
                ) * self.action_ranges - self.action_means

        else:
            value, disc_act, cont_act = self.Q1(observations, action_mask)
            # select actions from q function

        self.step += int(step)
        return 0, 0, 0  # Action 0, log_prob 0, value

    def ego_actions(self, observations, action_mask=None):
        return 0

    def imitation_learn(self, observations, actions):
        return 0  # loss

    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action):
        print("expected_V not implemeted")
        return 0

    def reinforcement_learn(self, batch, agent_num=0, critic_only=False, debug=False):
        return 0, 0  # actor loss, critic loss

    def save(self, checkpoint_path):
        print("Save not implemeted")

    def load(self, checkpoint_path):
        print("Load not implemented")
