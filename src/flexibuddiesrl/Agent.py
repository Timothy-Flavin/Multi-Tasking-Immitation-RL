from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Util import T


class Agent(ABC):
    from abc import ABC, abstractmethod


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Util import T


class Agent(ABC):

    @abstractmethod
    def train_actions(
        self, observations, action_mask=None, step=False, debug=False
    ) -> dict:
        return {
            "discrete_action": 0,
            "continuous_action": 0,
            "discrete_log_prob": 0,
            "continuous_log_prob": 0,
            "value": 0,
            "time": 0,
        }

    @abstractmethod
    def ego_actions(self, observations, action_mask=None) -> dict:
        return {
            "discrete_action": 0,
            "continuous_action": 0,
        }

    @abstractmethod
    def imitation_learn(
        self,
        observations,
        continuous_actions,
        discrete_actions,
        action_mask=None,
        debug=False,
    ) -> dict:
        immitation_metrics = {"critic_loss": 0, "actor_loss": 0, "time": 0}
        return immitation_metrics

    @abstractmethod
    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    @abstractmethod
    def expected_V(self, obs, legal_action) -> torch.Tensor | np.ndarray | float:
        print("expected_V not implemeted")
        return 0.0

    @abstractmethod
    def reinforcement_learn(
        self, batch, agent_num=0, critic_only=False, debug=False
    ) -> dict:
        rl_metrics = {
            "critic_loss": 0,
            "d_actor_loss": 0,
            "c_actor_loss": 0,
            "d_entropy": 0,
            "c_entropy": 0,
            "c_std": 0,
        }
        return rl_metrics

    @abstractmethod
    def save(self, checkpoint_path):
        print("Save not implemeted")

    @abstractmethod
    def load(self, checkpoint_path):
        print("Load not implemented")

    @abstractmethod
    def param_count(self) -> tuple[int, int]:
        return 0, 0  # train and execute param count


def _orthogonal_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ffEncoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dims,
        activation="relu",
        device="cpu",
        orthogonal_init=False,
        dropout=0.6,
    ):
        super(ffEncoder, self).__init__()
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "none": lambda x: x,
        }
        assert activation in activations, "Invalid activation function"
        self.activation = activations[activation]
        self.drop = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = nn.ModuleList()
        # print(obs_dim, hidden_dims)
        for i in range(len(hidden_dims)):
            if i == 0:
                self.encoder.append(nn.Linear(obs_dim, hidden_dims[i]))
            else:
                self.encoder.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if orthogonal_init:
                _orthogonal_init(self.encoder[-1])
        self.float()
        self.to(device)
        self.device = device
        # self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x, debug=False):
        if debug:
            print(f"ffEncoder: x {x}")
        x = T(x, self.device).float()
        if debug:
            print(f"ffEncoder after T: x {x}")
            interlist = []
            interlist.append(x)
        for layer in self.encoder:
            if layer == self.encoder[0] and self.drop > 0:
                x = self.activation(self.dropout(layer(x)))
            else:
                x = self.activation(layer(x))
            if debug:
                interlist.append(x)  # type: ignore
        # if x contains nan, print the intermediate list and encoder weights
        if torch.isnan(x).any():
            if debug:
                print(f"Intermediate list: {interlist}")  # type: ignore
            for layer in self.encoder:
                print(f"Layer {layer.weight}")  # type: ignore
        return x


class MixedActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=None,  # number of continuouis action dimensions =5
        discrete_action_dims=None,  # list of discrete action dimensions =[2, 3, 4]
        max_actions: np.ndarray = np.array([1.0], dtype=np.float32),
        min_actions: np.ndarray = np.array([-1.0], dtype=np.float32),
        hidden_dims: np.ndarray = np.array([256, 256], dtype=np.int32),
        encoder=None,  # ffEncoder if hidden dims are provided and encoder is not provided
        device="cpu",
        tau=1.0,
        hard=False,
        orthogonal_init=False,
        activation="relu",
    ):
        super(MixedActor, self).__init__()
        self.device = device

        self.tau = tau
        self.hard = hard
        # print(hidden_dims)
        if encoder is None and hidden_dims is not None and len(hidden_dims) > 0:
            self.encoder = ffEncoder(
                obs_dim, hidden_dims, device=device, activation=activation, dropout=0
            )

        assert not (
            continuous_action_dim is None and discrete_action_dims is None
        ), "At least one action dim should be provided"
        if continuous_action_dim is not None and continuous_action_dim > 0:
            assert (
                len(max_actions) == continuous_action_dim
                and len(min_actions) == continuous_action_dim
            ), f"max_actions should be provided for each continuous action dim {len(max_actions)},{continuous_action_dim}"

        # print(
        #    f"Min actions: {min_actions}, max actions: {max_actions}, torch {torch.from_numpy(max_actions - min_actions)}"
        # )
        if max_actions is not None and min_actions is not None:
            self.action_scales = (
                torch.from_numpy(max_actions - min_actions).float().to(device) / 2
            )
            # doesn't track grad by default in from_numpy
            self.action_biases = (
                torch.from_numpy(max_actions + min_actions).float().to(device) / 2
            )
            self.max_actions = max_actions
            self.min_actions = min_actions

        self.continuous_actions_head = None
        if continuous_action_dim is not None and continuous_action_dim > 0:
            self.continuous_actions_head = nn.Linear(
                hidden_dims[-1], continuous_action_dim
            )
            if orthogonal_init:
                _orthogonal_init(self.continuous_actions_head)

        self.discrete_action_heads = nn.ModuleList()
        if discrete_action_dims is not None and len(discrete_action_dims) > 0:
            for dim in discrete_action_dims:
                self.discrete_action_heads.append(nn.Linear(hidden_dims[-1], dim))
                if orthogonal_init:
                    _orthogonal_init(self.discrete_action_heads[-1])
        self.to(device)

    def forward(self, x, action_mask=None, gumbel=False, debug=False):
        ogx = x
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
            # If continuous action contains nan, print x and the continuous actions
            if torch.isnan(continuous_actions).any():
                print(f"Continuous actions: {continuous_actions}")
                print(f"X: {x}, ogx: {ogx}")
                # raise ValueError("Continuous actions contain nan")

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


class StochasticActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim: int = 0,  # number of continuous action dimensions = 4
        discrete_action_dims: (
            list[int] | None
        ) = None,  # list of discrete action dimensions =[2, 3, 4]
        max_actions: np.ndarray | None | torch.Tensor = np.array(
            [1.0], dtype=np.float32
        ),
        min_actions: np.ndarray | None | torch.Tensor = np.array(
            [-1.0], dtype=np.float32
        ),
        hidden_dims: np.ndarray | None | torch.Tensor = np.array(
            [64, 64], dtype=np.int32
        ),  # Last dim will be used to specify encoder output dim if one is supplied
        encoder=None,  # ffEncoder if hidden dims are provided and encoder is not provided
        device="cpu",
        gumbel_tau=1.0,  # for gumbel soft
        gumbel_tau_decay=0.9999,
        gumbel_tau_min=0.1,
        gumble_hard=False,
        orthogonal_init=False,
        activation="relu",  # activation function for the encoder
        action_head_hidden_dims=[32],  # iterable of hidden dims for action heads
        log_std_clamp_range=(-10, 2),
        std_type: str | None = None,  # full, diagonal, none
        clamp_type: str = "tanh",  # tanh, clamp, or None
    ):
        super(StochasticActor, self).__init__()
        self.encoder = None
        self.device = device
        self.std_type = std_type
        self.obs_dim = obs_dim
        if self.std_type is not None and self.std_type.lower() == "none":
            self.std_type = None
        self.clamp_type = clamp_type
        if clamp_type not in ["tanh", "clamp", None]:
            raise ValueError("clamp_type should be one of 'tanh', 'clamp', or None")

        self.gumbel_tau = gumbel_tau
        self.gumbel_tau_decay = gumbel_tau_decay
        self.gumbel_tau_min = gumbel_tau_min
        self.gumble_hard = gumble_hard
        self.continuous_action_dim = continuous_action_dim
        sizes = {None: 0, "full": continuous_action_dim, "diagonal": 1}
        self.log_std_dim = sizes[self.std_type]
        self.discrete_action_dims = discrete_action_dims
        self.log_std_clamp_range = log_std_clamp_range
        acts = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
        if activation in acts:
            self.activation = acts[activation]
        else:
            self.activation = activation

        self.provided_encoder = encoder is not None
        assert not (
            hidden_dims is None and encoder is None
        ), "If you do not provide an encoder then you need to provide mlp dims"

        # At this point, either encoder is not None or hidden_dims is not None (asserted above)
        if encoder is None and hidden_dims is not None and len(hidden_dims) > 0:
            self.encoder = ffEncoder(
                obs_dim,
                hidden_dims,
                device=device,
                activation=activation,
                dropout=0,
            )

        assert not (
            continuous_action_dim == 0 and discrete_action_dims is None
        ), "At least one action dim should be provided"
        if continuous_action_dim > 0:
            assert (
                max_actions is not None
                and min_actions is not None
                and len(max_actions) >= continuous_action_dim
                and len(min_actions) >= continuous_action_dim
            ), f"max_actions should be provided for each continuous action dim len(max): {len(max_actions) if max_actions is not None else None}, continuous_action_dim: {continuous_action_dim} min: {len(min_actions) if min_actions is not None else None}, continuous_action_dim: {continuous_action_dim}"

        # print(
        #    f"Min actions: {min_actions}, max actions: {max_actions}, torch {torch.from_numpy(max_actions - min_actions)}"
        # )
        if max_actions is not None and min_actions is not None:
            self.action_scales = (
                torch.from_numpy(max_actions - min_actions).float().to(device) / 2
            )
            # doesn't track grad by default in from_numpy
            self.action_biases = (
                torch.from_numpy(max_actions + min_actions).float().to(device) / 2
            )
            self.max_actions = torch.from_numpy(max_actions).to(device)
            self.min_actions = torch.from_numpy(min_actions).to(device)

        # Initialize the action head which outputs shape (continuous_action_dim * 2 + sum(discrete_action_dims))
        self._init_action_heads(
            hidden_dims,
            orthogonal_init,
            action_head_hidden_dims,
        )
        self.to(device)

    def _init_action_heads(
        self,
        hidden_dims,
        orthogonal_init,
        action_head_hidden_dims,
    ):
        self.action_layers = nn.ModuleList()

        if self.provided_encoder:
            last_hidden_dim = self.obs_dim
        else:
            last_hidden_dim = hidden_dims[-1]
        assert (
            self.std_type.lower()
            if self.std_type is not None
            else None
            in [
                None,
                "diagonal",
                "full",
            ]
        ), "standard deviation type must be chosen from [None,'none','diagonal','full'] by setting std_type=..."

        # Setting output dimension depending on what kind of continuous value is getting pulled
        output_dim = 0
        if self.discrete_action_dims is not None:
            output_dim = sum(self.discrete_action_dims)
        output_dim += self.continuous_action_dim + self.log_std_dim

        # setting up layers for action dims. These need to be independend becaue
        if action_head_hidden_dims is not None and len(action_head_hidden_dims) > 0:
            for i, dim in enumerate(action_head_hidden_dims):
                if i == 0:
                    self.action_layers.append(nn.Linear(last_hidden_dim, dim))
                else:
                    self.action_layers.append(
                        nn.Linear(action_head_hidden_dims[i - 1], dim)
                    )

                if orthogonal_init:
                    _orthogonal_init(self.action_layers[-1])
            last_hidden_dim = action_head_hidden_dims[-1]
        # This needs to stay outside that if so there is always an action head, just not a multilayer one
        self.action_layers.append(nn.Linear(last_hidden_dim, output_dim))
        if orthogonal_init:
            _orthogonal_init(self.action_layers[-1])

    # TODO: action mask implementation
    def forward(self, x, action_mask=None, debug=False):
        if debug:
            print(f"  MixedActor forward: x {x}, action_mask {action_mask}, ")

        embedding = self.encoder(x=x, debug=debug) if self.encoder is not None else x
        for i, layer in enumerate(self.action_layers):
            if i != len(self.action_layers) - 1:
                embedding = F.relu(layer(embedding))
            else:
                embedding = layer(embedding)
            if debug:
                print(f"  MixedActor: embedding shape {embedding.shape}")

        # if embedding is a single vector, unsqueeze it to make it a batch of size 1
        single_vector = False
        if len(embedding.shape) == 1:
            single_vector = True
            embedding = embedding.unsqueeze(0)

        continuous_means = None
        continuous_log_std_logits = None
        discrete_logits = None

        # Get the continuous means and log std logits id std_type is not None
        if self.continuous_action_dim > 0:
            continuous_means = embedding[:, : self.continuous_action_dim]
            continuous_log_std_logits = None
            if self.log_std_dim > 0:
                continuous_log_std_logits = embedding[
                    :,
                    self.continuous_action_dim : self.continuous_action_dim
                    + self.log_std_dim,
                ]
                if self.clamp_type == "tanh":
                    continuous_log_std_logits = torch.tanh(continuous_log_std_logits)
                    continuous_log_std_logits = self.log_std_clamp_range[0] + 0.5 * (
                        self.log_std_clamp_range[1] - self.log_std_clamp_range[0]
                    ) * (
                        continuous_log_std_logits + 1
                    )  # From CleanRL / SpinUp / Denis Yarats
                else:  # self.clamp_type == "clamp" Always clamp at least
                    continuous_log_std_logits = torch.clamp(
                        continuous_log_std_logits,
                        self.log_std_clamp_range[0],
                        self.log_std_clamp_range[1],
                    )

        if debug:
            print(
                f"  MixedActor: continuous_means shape {continuous_means.shape if continuous_means is not None else None}, "
                f"  continuous_log_std_logits shape {continuous_log_std_logits.shape if continuous_log_std_logits is not None else None}"
            )
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            discrete_logits: list[torch.Tensor] | None = []
            start = self.continuous_action_dim + self.log_std_dim
            for i, dim in enumerate(self.discrete_action_dims):
                end = start + dim
                logits = embedding[:, start:end]
                discrete_logits.append(logits)
                # print(logits)

        if single_vector:
            continuous_means = (
                continuous_means.squeeze(0) if continuous_means is not None else None
            )
            continuous_log_std_logits = (
                continuous_log_std_logits.squeeze(0)
                if continuous_log_std_logits is not None
                else None
            )
            discrete_logits = (
                [logits.squeeze(0) for logits in discrete_logits]
                if discrete_logits is not None
                else None
            )
        return continuous_means, continuous_log_std_logits, discrete_logits

    # TODO refactor into this
    def sample_continuous(self, means, log_stds, with_log_probs=False):
        pass

    def sample_discrete(self, logits, with_log_probs=False):
        pass

    # DDPG way
    def deterministic_action(self, c_logits, d_logits, noise_generator, gumble=False):
        pass

    # PPO / SAC way
    def stochastic_action(
        self,
        c_logits,
        d_logits,
        c_log_stds,
        with_logc=False,
        with_logd=False,
        gumble=False,
    ):
        pass

    def action_from_logits(
        self,
        continuous_means: torch.Tensor | None,
        continuous_log_std_logits: torch.Tensor | None,
        discrete_logits: list[torch.Tensor] | None,
        gumble: bool = False,
        log_con: bool = False,
        log_disc: bool = False,
    ) -> tuple[
        None | torch.Tensor | list[torch.Tensor],
        None | torch.Tensor | list[torch.Tensor],
        None | torch.Tensor | list[torch.Tensor],
        None | torch.Tensor | list[torch.Tensor],
        None | torch.Tensor | list[torch.Tensor],
    ]:
        """
        Produces actions from logits with gradient maintained always for continuous and if gumbel is True for discrete
        Args:
            continuous_means: (batch_size, continuous_action_dim)
            continuous_log_std_logits: (batch_size, continuous_action_dim)
            discrete_logits: list of (batch_size, discrete_action_dim[i]) for each discrete action dim
        Returns:
            continuous_actions: (batch_size, continuous_action_dim)
                Continuous actions are sampled from a normal distribution with means and std parameterized before
                passing through tanh and scaled to the action space
            discrete_actions:
                If Gumbel: list of len n [(batch_size, discrete_action_dim[i])] for each discrete action dim 'i' in 'n'
                Else: torch.Long shape = (batch_size, len(discrete_action_dims))

                Gumbel softmax is to retain the gradient through reparameterization trick so it is like returning
                a soft one-hot coding with a tensor for each discrete action dim
                If not gumbel, then it is a long tensor of the sampled actions where each action is sampled from
                the categorical distribution
        """
        continuous_actions = None
        discrete_actions = None
        continuous_log_probs = None
        discrete_log_probs = None
        continuous_activations = None
        c_dist = None
        d_dist = None

        assert not (
            continuous_log_std_logits is None and log_con
        ), f"clstdl: {continuous_log_std_logits}, log_con: {log_con} You can't get log probs from just a mean, log stds was none"

        if self.continuous_action_dim > 0:
            assert (
                continuous_means is not None
            ), "Cant have continuous dims with no logits"
            if continuous_log_std_logits is None:
                continuous_activations = continuous_means
            else:
                # if full do it this way
                # If log_std is shape (m, 1), expand to (m, n) to match means
                log_std = continuous_log_std_logits
                # if log_std.shape[1] == 1 and continuous_means.shape[1] > 1:
                log_std = log_std.expand_as(continuous_means)
                c_dist = torch.distributions.Normal(
                    continuous_means,
                    torch.exp(log_std),
                )
                continuous_activations = c_dist.rsample()

            if self.clamp_type == "tanh":
                continuous_actions = (
                    torch.tanh(continuous_activations)  # torch.clamp()?
                    * self.action_scales
                    + self.action_biases
                )
                if log_con:
                    assert (
                        c_dist is not None
                    ), "Somehow we want log probs from a distirbution that doesn't exist"

                    # This is from spinningup SAC
                    continuous_log_probs = c_dist.log_prob(continuous_activations).sum(
                        axis=-1
                    )
                    continuous_log_probs -= (
                        2
                        * (
                            np.log(2)
                            - continuous_activations
                            - F.softplus(-2 * continuous_activations)
                        )
                    ).sum(axis=-1)
                # print(
                #     f"Continuous actions: {continuous_actions}, log probs: {continuous_log_probs} continuous_activations: {continuous_activations}, means: {continuous_means}, stds: {torch.exp(continuous_log_std_logits) if continuous_log_std_logits is not None else None}"
                # )
            elif self.clamp_type == "clamp":
                continuous_actions = torch.clamp(
                    continuous_activations, self.min_actions, self.max_actions
                )
                if log_con:
                    assert (
                        c_dist is not None
                    ), "Somehow we want log probs from a distirbution that doesn't exist"
                    continuous_log_probs = c_dist.log_prob(continuous_activations).sum(
                        axis=-1
                    )
            else:
                continuous_actions = continuous_activations
                if log_con:
                    assert (
                        c_dist is not None
                    ), "Somehow we want log probs from a distirbution that doesn't exist"
                    # print(c_dist.log_prob(continuous_activations))
                    continuous_log_probs = c_dist.log_prob(continuous_activations).sum(
                        axis=-1
                    )
            # print(
            #    f"{self.clamp_type} Continuous actions: {continuous_actions}, log probs: {continuous_log_probs} from  {c_dist.log_prob(continuous_activations) if c_dist is not None else None} continuous_activations: {continuous_activations}, means: {continuous_means}, stds: {torch.exp(continuous_log_std_logits) if continuous_log_std_logits is not None else None}"
            # )
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            assert (
                discrete_logits is not None
            ), "Cant have discrete action dim and no discrete actions"
            if gumble:
                discrete_actions: list[torch.Tensor] | torch.Tensor | None = []
                for i, logits in enumerate(discrete_logits):
                    probs = F.gumbel_softmax(
                        logits, dim=-1, tau=self.gumbel_tau, hard=self.gumble_hard
                    )
                    discrete_actions.append(probs)
            else:
                if len(discrete_logits[0].shape) == 1:
                    discrete_actions = torch.zeros(
                        len(self.discrete_action_dims),
                        device=self.device,
                        dtype=torch.long,
                    )
                    if log_disc:
                        discrete_log_probs = torch.zeros(
                            len(self.discrete_action_dims), device=self.device
                        )
                else:
                    discrete_actions = torch.zeros(
                        discrete_logits[0].shape[0],
                        len(self.discrete_action_dims),
                        device=self.device,
                        dtype=torch.long,
                    )
                    if log_disc:
                        discrete_log_probs = torch.zeros(
                            discrete_logits[0].shape[0],
                            len(self.discrete_action_dims),
                            device=self.device,
                        )
                for i, logits in enumerate(discrete_logits):
                    if len(discrete_logits[0].shape) == 1:
                        d_dist = torch.distributions.Categorical(logits=logits)
                        discrete_actions[i] = d_dist.sample()
                        if log_disc and discrete_log_probs is not None:
                            try:
                                discrete_log_probs[i] = d_dist.log_prob(
                                    discrete_actions[i]
                                )
                            except Exception as e:
                                print(discrete_actions[i])
                                print(logits)
                                print("hmmmm")
                                raise e
                    else:
                        d_dist = torch.distributions.Categorical(logits=logits)
                        discrete_actions[:, i] = d_dist.sample()
                        if log_disc and discrete_log_probs is not None:
                            discrete_log_probs[:, i] = d_dist.log_prob(
                                discrete_actions[:, i]
                            )
        return (
            discrete_actions,
            continuous_actions,
            discrete_log_probs,
            continuous_log_probs,
            continuous_activations,
        )


class ValueSA(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dim=256, device="cpu", activation="relu"
    ):
        super(ValueSA, self).__init__()
        self.device = device
        if activation not in ["relu", "tanh", "sigmoid"]:
            raise ValueError(
                "Invalid activation function, should be: relu, tanh, sigmoid"
            )
        activations = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
        self.activation = activations[activation]
        self.l1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, x, u, debug=False):
        if debug:
            print(f"ValueSA: x {x}, u {u}")
        x = self.activation(self.l1(torch.cat([x, u], -1)))
        x = self.activation(self.l2(x))
        x = self.l3(x)
        return x


class ValueS(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim=256,
        device="cpu",
        activation="relu",
        orthogonal_init=False,
    ):
        super(ValueS, self).__init__()
        self.device = device
        if activation not in ["relu", "tanh", "sigmoid"]:
            raise ValueError(
                "Invalid activation function, should be: relu, tanh, sigmoid"
            )
        activations = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
        self.activation = activations[activation]
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        if orthogonal_init:
            _orthogonal_init(self.l1)
            _orthogonal_init(self.l2)
            _orthogonal_init(self.l3)
        self.to(device)

    def forward(self, x):
        x = T(x, self.device)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.l3(x)
        return x


class QSCA(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[1],
        hidden_dim=256,
        device="cpu",
    ):
        super(QSCA, self).__init__()
        self.device = device
        self.l1 = nn.Linear(obs_dim + continuous_action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.discrete_Q_heads = nn.ModuleList()
        if discrete_action_dims is not None and len(discrete_action_dims) > 0:
            for dim in discrete_action_dims:
                self.discrete_Q_heads.append(nn.Linear(hidden_dim, dim))
        self.to(device)

    def forward(self, x):
        x = T(x, self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        Qs = []
        for i, head in enumerate(self.discrete_Q_heads):
            Qi = head(x)
            Qs.append(Qi)
        if len(Qs) == 1:
            Qs = Qs[0]
        return Qs


class QSAA(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[1],
        hidden_dim=256,
        device="cpu",
    ):
        super(QSAA, self).__init__()
        self.device = device
        total_discrete_dims = sum(discrete_action_dims)
        input_dim = obs_dim + continuous_action_dim + total_discrete_dims
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, s, a_c=None, a_d=None):
        if a_c is None:
            a_c = torch.tensor([]).to(self.device)
        if a_d is None:
            a_d = torch.tensor([]).to(self.device)
        x = torch.cat([s, a_c, a_d], dim=-1)
        x = T(x, self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class QMixer(nn.Module):
    """
    A 2-layer monotonic mixing network, as used in QMIX.

    It takes the Q-values of individual agents and the global state as input
    and outputs a single total Q-value for the team. The mixing network's
    weights are generated by a hypernetwork conditioned on the global state.
    Monotonicity is enforced by constraining these weights to be non-negative.
    """

    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 64):
        """
        Initializes the QMixer network.

        Args:
            n_agents (int): The number of agents in the team.
            state_dim (int): The dimension of the global state.
            mixing_embed_dim (int): The dimension of the mixing network's hidden layer.
        """
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # Hypernetwork for the first mixing layer's weights
        # It generates a weight matrix of shape (n_agents, mixing_embed_dim)
        self.hyper_w1 = nn.Linear(self.state_dim, self.n_agents * self.embed_dim)

        # Hypernetwork for the first mixing layer's bias
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)

        # Hypernetwork for the second mixing layer's weights
        # It generates a weight vector of shape (mixing_embed_dim, 1)
        self.hyper_w2 = nn.Linear(self.state_dim, self.embed_dim)

        # State-dependent bias for the final output (V(s) term)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor, with_grad: bool = False
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        """
        Forward pass for the QMixer.

        Args:
            agent_qs (torch.Tensor): The Q-values of individual agents.
                                     Shape: (batch_size, n_agents)
            state (torch.Tensor): The global state.
                                  Shape: (batch_size, state_dim)

        Returns:
            torch.Tensor: The total Q-value for the team.
                          Shape: (batch_size, 1)
        """
        batch_size = agent_qs.size(0)

        # --- Generate weights and biases from the state using hypernetworks ---

        # First layer weights and biases
        # Enforce non-negativity on weights for monotonicity
        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)

        # Second layer weights and the final bias (V(s))
        w2 = torch.abs(self.hyper_w2(state))
        b2 = self.hyper_b2(state)

        # --- Reshape for batch matrix multiplication ---

        # Reshape weights and biases to match linear layer dimensions
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = b1.view(batch_size, 1, self.embed_dim)

        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = b2.view(batch_size, 1, 1)

        # Reshape agent Q-values for mixing
        agent_qs_view = agent_qs.view(batch_size, 1, self.n_agents)

        # --- Perform the mixing ---

        # First mixing layer
        # Use ELU activation, as is common in QMIX implementations
        hidden = F.elu(torch.bmm(agent_qs_view, w1) + b1)

        # Second mixing layer
        q_total = torch.bmm(hidden, w2) + b2

        q_grads = None
        if with_grad:
            q_total.sum().backward()
            q_grads = agent_qs.grad

        return q_total.view(batch_size, -1), q_grads


class VDNMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super(VDNMixer, self).__init__()

    def forward(
        self, agent_qs: torch.Tensor, state: torch.Tensor, with_grad: bool = False
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        return agent_qs.sum(dim=-1, keepdim=True), torch.ones_like(agent_qs)


class QS(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[2],
        hidden_dims=[64, 64],
        encoder=None,
        activation="relu",
        orthogonal=False,
        dropout=0.0,
        dueling=False,
        device="cpu",
        n_c_action_bins=11,
        head_hidden_dims=[32],  # [64],
        verbose=False,
        QMIX=True,
        QMIX_hidden_dim=64,
    ):
        super(QS, self).__init__()

        self.QMIX = QMIX
        # guard code
        if continuous_action_dim is None:
            continuous_action_dim = 0

        assert (
            encoder is not None or hidden_dims is not None
        ), "Either an encoder must be supplied, or hidden dims so the model will make a feed forward encoder"
        self.encoder = None
        if encoder is not None:
            self.encoder = encoder
        elif hidden_dims is not None:
            self.encoder = ffEncoder(
                obs_dim, hidden_dims, activation, device, orthogonal, dropout
            )

        if discrete_action_dims is not None:
            if isinstance(discrete_action_dims, int):
                discrete_action_dims = [discrete_action_dims]
            if len(discrete_action_dims) == 0:
                ValueError(
                    "discrete_action_dims should not be empty, use [x] for a single discrete action with cardonality 'x'"
                )
            if min(discrete_action_dims) < 1:
                ValueError(
                    "discrete_action_dims should not contain values less than 1, use [x] for a single discrete action with cardonality 'x'"
                )
        self.last_hidden_dim = (
            obs_dim  # this will be the encoder output dim if encoder is supplied
        )
        if encoder is None:
            self.last_hidden_dim = hidden_dims[-1]

        self.mixing_network = None
        if self.QMIX:
            qdim = (
                len(discrete_action_dims) if discrete_action_dims is not None else 0
            ) + continuous_action_dim
            self.mixing_network = QMixer(qdim, self.last_hidden_dim, QMIX_hidden_dim)

        # setting needed self variables
        self.disc_action_dims = discrete_action_dims
        self.cont_action_dim = continuous_action_dim
        self.n_heads = continuous_action_dim  # add discrete under non guard
        self.device = device
        self.dueling = dueling

        self.tot_adv_size = continuous_action_dim * n_c_action_bins
        if discrete_action_dims is not None:
            self.tot_adv_size += sum(discrete_action_dims)
            self.n_heads += len(discrete_action_dims)

        self.value_dim = 1
        if self.dueling:
            # if self.value_per_head:
            #     self.value_dim = self.n_heads
            # else:
            self.value_dim = 1
        # set up hidden layer for the adv and V heads
        joint_head_layers = []
        if head_hidden_dims is not None:
            # print(head_hidden_dims)
            head_hidden_dims[-1] = head_hidden_dims[-1] * (
                2 if dueling else 1  # separate embeddings if dueling
            )  # need independent chunk for value vs advantage
            for i, dim in enumerate(head_hidden_dims):
                joint_head_layers.append(
                    nn.Linear(
                        self.last_hidden_dim if i == 0 else head_hidden_dims[i - 1],
                        dim,
                    )
                )
                if orthogonal:
                    _orthogonal_init(joint_head_layers[-1])
            self.last_hidden_dim = head_hidden_dims[-1]
            self.joint_head_layers = nn.ModuleList(joint_head_layers)
        else:
            self.joint_head_layers = None

        # set up the adv heads if this isn't just a V network
        if self.cont_action_dim > 0 or self.disc_action_dims is not None:
            self.advantage_heads = nn.Linear(
                self.last_hidden_dim // (2 if dueling else 1),  #
                self.tot_adv_size,
            )

        # set up the value head if dueling is True
        if self.dueling:
            # if self.value_per_head:
            #     self.value_head = nn.Linear(
            #         self.last_hidden_dim // 2,
            #         (
            #             len(discrete_action_dims)
            #             if discrete_action_dims is not None
            #             else 0
            #         )
            #         + continuous_action_dim,  # because the continuous dims are discrete behind the hood
            #     )
            # else:
            self.value_head = nn.Linear(self.last_hidden_dim // 2, 1)
        else:
            self.value_head = None
        if verbose:
            print(
                f"initialized QS with: {self.joint_head_layers}, {self.value_head}, {self.advantage_heads}\n  d_dim: {discrete_action_dims}, c_dim: {continuous_action_dim}, h_dim: {hidden_dims}, head_hidden_dims: {head_hidden_dims}"
            )
        self.to(device)

    def factorize_Q(self, qs, state):
        if self.encoder is not None:
            state = self.encoder(state)
        assert (
            self.mixing_network is not None
        ), "Cant qmix factorize q values if the mixing network has not been initialized"
        return self.mixing_network(qs, state)

    def forward(self, x, action_mask=None):
        # TODO: action mask implementation
        x = T(x, self.device)
        # print(f"starting x shape: {x.shape} {len(x.shape)}")
        if self.encoder is not None:
            x = self.encoder(x)
        values = 0

        single_dim = False
        # print(f" x shape: {x.shape} {len(x.shape)}")
        if len(x.shape) == 1:
            single_dim = True
            x = x.unsqueeze(0)

        # If the heads have their own hidden layers for a 2 layer dueling network
        if self.joint_head_layers is not None:
            for li in self.joint_head_layers:
                x = torch.tanh(li(x))
        if self.dueling and self.value_head is not None:
            values = self.value_head(x[:, : self.last_hidden_dim // 2])
            if single_dim:
                values = values.squeeze(0)

        # half the embedding belongs to the value head if dueling
        advantages = None
        if self.dueling and self.advantage_heads is not None:
            advantages = self.advantage_heads(x[:, -self.last_hidden_dim // 2 :])
        elif self.advantage_heads is not None:
            advantages = self.advantage_heads(x)

        tot_disc_dims = 0
        disc_advantages = None
        cont_advantages = None
        if self.disc_action_dims is not None and advantages is not None:
            tot_disc_dims = sum(self.disc_action_dims)
            disc_advantages = []
            start = 0
            for i, dim in enumerate(self.disc_action_dims):
                end = start + dim
                disc_advantages.append(advantages[:, start:end])
                if (
                    self.dueling
                ):  # These are mean zero when dueling or Q values when not
                    disc_advantages[-1] = disc_advantages[-1] - disc_advantages[
                        -1
                    ].mean(dim=-1, keepdim=True)
                if single_dim:
                    disc_advantages[-1] = disc_advantages[-1].squeeze(0)
                start = end

        if self.cont_action_dim > 0 and advantages is not None:
            cont_advantages = (
                advantages[:, tot_disc_dims:].view(
                    advantages.shape[0], self.cont_action_dim, -1
                )
                # .transpose(0, 1)  # TODO: figure out if it is worth it to transpose
            )  # transposed because then discrete and continuous output same dim order
            if self.dueling:  # These are mean zero when dueling or Q values when not
                cont_advantages = cont_advantages - cont_advantages.mean(
                    dim=-1, keepdim=True
                )
            if single_dim:
                # print(f"single dim: {cont_advantages.shape}")
                cont_advantages = cont_advantages.squeeze(0)

        return values, disc_advantages, cont_advantages


class DuelingQSCA(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[1],
        hidden_dim=256,
        device="cpu",
    ):
        super(DuelingQSCA, self).__init__()
        self.device = device
        self.l1 = nn.Linear(obs_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_heads = nn.ModuleList()
        if discrete_action_dims is not None and len(discrete_action_dims) > 0:
            for dim in discrete_action_dims:
                self.advantage_heads.append(
                    nn.Linear(hidden_dim + continuous_action_dim, dim)
                )
        self.to(device)

    def forward(self, x, u):
        x = T(x, self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        values = self.value_head(x)
        advantages = []
        xu = torch.cat([x, u], dim=-1)
        for i, head in enumerate(self.advantage_heads):
            Adv = head(xu)
            Adv = Adv - Adv.mean(dim=-1, keepdim=True)
            advantages.append(Adv)
        return values, advantages


class DuelingQSAA(nn.Module):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=[1],
        hidden_dim=256,
        device="cpu",
    ):
        super(DuelingQSAA, self).__init__()
        self.device = device
        total_discrete_dims = sum(discrete_action_dims)
        input_dim = obs_dim
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_heads = [
            nn.Linear(hidden_dim + continuous_action_dim + total_discrete_dims, 1)
        ]

        self.to(device)

    def forward(self, x, a_c=None, a_d=None):
        if a_c is None:
            a_c = torch.tensor([]).to(self.device)
        if a_d is None:
            a_d = torch.tensor([]).to(self.device)

        x = T(x, self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        values = self.value_head(x)
        advantages = []
        xu = torch.cat([x, a_c, a_d], dim=-1)
        for i, head in enumerate(self.advantage_heads):
            Adv = head(xu)
            # Adv = Adv - Adv.mean(dim=-1, keepdim=True)
            # Sample some kind of action space and then calculate the advantage
            advantages.append(Adv)
        return values, advantages


# Q(s) -> R^n
# Q(s,a) -> R

# Q(s,a_c)     = R^n
# Q(s,a_c,a_d) = R

# Q(s) ->   V(s)+A(s,a)      A - mean(A)
# Q(s,a) -> V(s)+A(s,a)      Posisibilities to adapt


if __name__ == "__main__":
    device = "cuda"
    # Example instantiations
    c_dim = 2
    d_dims = [3, 4]
    actor = MixedActor(
        obs_dim=10,
        continuous_action_dim=c_dim,
        discrete_action_dims=d_dims,
        max_actions=np.array([1.0, 1.0]),
        min_actions=np.array([-1.0, -1.0]),
        hidden_dims=np.array([256, 256]),
        device=device,
    )

    value_sa = ValueSA(
        obs_dim=10, action_dim=c_dim + np.sum(d_dims), hidden_dim=256, device=device
    )

    value_s = ValueS(obs_dim=10, hidden_dim=256, device=device)

    q_net = QSCA(
        obs_dim=10,
        hidden_dim=256,
        discrete_action_dims=d_dims,
        continuous_action_dim=c_dim,
        device=device,
    )
    qsaa_net = QSAA(
        obs_dim=10,
        continuous_action_dim=c_dim,
        discrete_action_dims=d_dims,
        hidden_dim=256,
        device=device,
    )
    state = torch.rand(size=(10,)).to(device)
    states = torch.rand(size=(5, 10)).to(device)

    # Single state through actor
    cont_acts, disc_acts = actor(state, gumbel=True)
    print("\nSingle state through actor:")
    print(
        "Continuous actions:",
        cont_acts,
        "Shape:",
        cont_acts.shape if cont_acts is not None else None,
    )
    for i, da in enumerate(disc_acts):
        print(
            f"Discrete action {i}:", da, "Shape:", da.shape if da is not None else None
        )

    # Batch of states through actor
    cont_acts_batch, disc_acts_batch = actor(states)
    print("\nBatch of states through actor:")
    print(
        "Continuous actions:",
        cont_acts_batch,
        "Shape:",
        cont_acts_batch.shape if cont_acts_batch is not None else None,
    )
    for i, da in enumerate(disc_acts_batch):
        print(
            f"Discrete action {i}:", da, "Shape:", da.shape if da is not None else None
        )

    print("Discrete Actions Concatenated")
    print(torch.cat(disc_acts, dim=0))

    print("All actions concatenated")
    print(torch.cat((cont_acts, torch.cat(disc_acts, dim=-1)), dim=-1))
    # Test value functions
    # Single state
    val_sa_out = value_sa(
        state, torch.cat((cont_acts, torch.cat(disc_acts, dim=-1)), dim=-1)
    )
    val_s_out = value_s(state)
    q_out = q_net(torch.cat((state, cont_acts), dim=-1))
    # Test single state through QSAA
    qsaa_out = qsaa_net(state, cont_acts, torch.cat(disc_acts, dim=-1))

    print("\nSingle state through value networks:")
    print("ValueSA output:", val_sa_out, "Shape:", val_sa_out.shape)
    print("ValueS output:", val_s_out, "Shape:", val_s_out.shape)
    print(
        "Q output:",
        q_out,
        "Shape:",
        (
            [q.shape if isinstance(q, torch.Tensor) else None for q in q_out]
            if isinstance(q_out, list)
            else q_out.shape
        ),
    )
    print("QSAA batch: ", qsaa_out, "Shape: ", qsaa_out.shape)

    print("Discrete Actions Batch Concatenated")
    print(torch.cat(disc_acts_batch, dim=-1))

    print("All actions Batch concatenated")
    print(torch.cat((cont_acts_batch, torch.cat(disc_acts_batch, dim=-1)), dim=-1))
    # Batch of states
    val_sa_batch = value_sa(
        states, torch.cat((cont_acts_batch, torch.cat(disc_acts_batch, dim=-1)), dim=-1)
    )
    val_s_batch = value_s(states)
    q_batch = q_net(torch.cat((states, cont_acts_batch), dim=-1))
    # Test batch of states through QSAA
    qsaa_batch = qsaa_net(states, cont_acts_batch, torch.cat(disc_acts_batch, dim=-1))

    print("\nBatch of states through value networks:")
    print("ValueSA batch output:", val_sa_batch, "Shape:", val_sa_batch.shape)
    print("ValueS batch output:", val_s_batch, "Shape:", val_s_batch.shape)
    print(
        "Q batch output:",
        q_batch,
        "Shape:",
        (
            [q.shape if isinstance(q, torch.Tensor) else None for q in q_batch]
            if isinstance(q_batch, list)
            else q_batch.shape
        ),
    )
    print("QSAA batch: ", qsaa_batch, "Shape: ", qsaa_batch.shape)
