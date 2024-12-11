from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Util import T


class Agent(ABC):

    @abstractmethod
    def train_actions(self, observations, action_mask=None, step=False):
        return 0, 0, 0  # Action 0, log_prob 0, value

    @abstractmethod
    def ego_actions(self, observations, action_mask=None):
        return 0

    @abstractmethod
    def imitation_learn(self, observations, actions):
        return 0  # loss

    @abstractmethod
    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    @abstractmethod
    def expected_V(self, obs, legal_action):
        print("expected_V not implemeted")
        return 0

    @abstractmethod
    def reinforcement_learn(self, batch, agent_num=0, critic_only=False, debug=False):
        return 0, 0  # actor loss, critic loss

    @abstractmethod
    def save(self, checkpoint_path):
        print("Save not implemeted")

    @abstractmethod
    def load(self, checkpoint_path):
        print("Load not implemented")


class ffEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dims, activation="relu", device="cpu"):
        super(ffEncoder, self).__init__()
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "none": lambda x: x,
        }
        assert activation in activations, "Invalid activation function"
        self.activation = activations[activation]
        self.encoder = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.encoder.append(nn.Linear(obs_dim, hidden_dims[i]))
            else:
                self.encoder.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x, debug=False):
        if debug:
            print(f"ffEncoder: x {x}")
        x = T(x, self.device)
        if debug:
            print(f"ffEncoder after T: x {x}")
        for layer in self.encoder:
            x = self.activation(layer(x))
        return x


class MixedActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=None,  # number of continuouis action dimensions =5
        discrete_action_dims=None,  # list of discrete action dimensions =[2, 3, 4]
        max_actions: np.array = np.array([1.0], dtype=np.float32),
        min_actions: np.array = np.array([-1.0], dtype=np.float32),
        hidden_dims: np.array = np.array([256, 256], dtype=np.int32),
        encoder=None,  # ffEncoder if hidden dims are provided and encoder is not provided
        device="cpu",
        tau=1.0,
        hard=False,
    ):
        super(MixedActor, self).__init__()
        self.device = device

        self.tau = tau
        self.hard = hard

        if encoder is None and len(hidden_dims) > 0:
            self.encoder = ffEncoder(obs_dim, hidden_dims, device=device)

        assert not (
            continuous_action_dim is None and discrete_action_dims is None
        ), "At least one action dim should be provided"
        assert len(max_actions) == len(discrete_action_dims) and len(
            min_actions
        ) == len(
            discrete_action_dims
        ), "max_actions should be provided for each discrete action dim"

        print(
            f"Min actions: {min_actions}, max actions: {max_actions}, torch {torch.from_numpy(max_actions - min_actions)}"
        )
        if max_actions is not None and min_actions is not None:
            self.action_scales = (
                torch.from_numpy(max_actions - min_actions).float().to(device) / 2
            )
            # doesn't track grad by default in from_numpy
            self.action_biases = (
                torch.from_numpy(max_actions + min_actions).float().to(device) / 2
            )

        self.continuous_actions_head = None
        if continuous_action_dim is not None and continuous_action_dim > 0:
            self.continuous_actions_head = nn.Linear(
                hidden_dims[-1], continuous_action_dim
            )

        self.discrete_action_heads = nn.ModuleList()
        if discrete_action_dims is not None and len(discrete_action_dims) > 0:
            for dim in discrete_action_dims:
                self.discrete_action_heads = nn.ModuleList().append(
                    nn.Linear(hidden_dims[-1], dim)
                )
        self.max_actions = max_actions

    def forward(self, x, action_mask=None, gumbel=False, debug=False):
        if debug:
            print(f"MixedActor: x {x}, action_mask {action_mask}, gumbel {gumbel}")
        if self.encoder is not None:
            x = self.encoder(x=x, debug=debug)
        else:
            x = T(a=x, device=self.device, debug=debug)

        continuous_actions = None
        discrete_actions = None
        if self.continuous_actions_head is not None:
            continuous_actions = (
                F.tanh(self.continuous_actions_head(x)) * self.action_scales
                + self.action_biases
            )

        # TODO: Put this into it's own function and implement the ppo way of sampling
        if self.discrete_action_heads is not None:
            discrete_actions = []
            for i, head in enumerate(self.discrete_action_heads):
                logits = head(x)

                if gumbel:
                    if action_mask is not None:
                        logits[action_mask == 0] = -1e8
                    probs = F.gumbel_softmax(
                        logits, dim=-1, tau=self.tau, hard=self.hard
                    )
                    # activations = activations / activations.sum(dim=-1, keepdim=True)
                    discrete_actions.append(probs)
                else:
                    if action_mask is not None:
                        logits[action_mask == 0] = -1e8
                    discrete_actions.append(F.softmax(logits, dim=-1))

        return continuous_actions, discrete_actions


class ValueSA(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, device="cpu"):
        super(ValueSA, self).__init__()
        self.device = device
        self.l1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, x, u, debug=False):
        if debug:
            print(f"ValueSA: x {x}, u {u}")
        x = F.relu(self.l1(torch.cat([x, u], -1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class ValueS(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, device="cpu"):
        super(ValueS, self).__init__()
        self.device = device
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, x):
        x = T(x, self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
