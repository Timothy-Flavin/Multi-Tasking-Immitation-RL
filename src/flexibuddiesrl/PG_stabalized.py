from .Agent import ValueS, StochasticActor, Agent, QMixer, QS, VDNMixer
from .Util import T, minmaxnorm
import torch
from flexibuff import FlexiBatch, FlexibleBuffer
from torch.distributions import Categorical
import numpy as np
import torch.nn as nn
import pickle
import os
import time
from torch.distributions import TransformedDistribution, TanhTransform
import torch.nn.functional as F
import collections


class PG(nn.Module, Agent):
    def __init__(
        self,
        obs_dim=10,
        continuous_action_dim=0,
        max_actions=None,
        min_actions=None,
        discrete_action_dims=None,
        lr=1e-4,
        gamma=0.99,
        n_epochs=2,
        device="cpu",
        entropy_loss=0.05,
        hidden_dims=[256, 256],
        activation="relu",
        ppo_clip=0.2,
        value_loss_coef=0.5,
        value_clip=0.5,
        advantage_type="gae",  # [g, gv, a2c, constant, gae, qmix]
        norm_advantages=True,
        mini_batch_size=64,
        anneal_lr=200000,
        orthogonal=True,
        clip_grad=True,
        gae_lambda=0.95,
        load_from_checkpoint=None,
        name="PPO",
        eval_mode=False,
        encoder=None,
        action_head_hidden_dims=None,
        std_type="stateless",  # ['full' 'diagonal' or 'stateless']
        naive_imitation=False,  # if true, do MSE instead of MLE
        action_clamp_type="tanh",
        batch_name_map={
            "discrete_actions": "discrete_actions",
            "continuous_actions": "continuous_actions",
            "rewards": "global_rewards",
            "obs": "obs",
            "obs_": "obs_",
            "continuous_log_probs": "continuous_log_probs",
            "discrete_log_probs": "discrete_log_probs",
            "truncated": "truncated",
            "terminated": "terminated",
        },
        mix_type=None,  # [None, 'VDN', 'QMIX']
        mixer_dim=128,
        importance_schedule=[10.0, 1.0, 10000],  # start end nsteps
        importance_from_grad=True,
        softmax_importance_scale=True,
        on_policy_mixer=True,
        logit_reg=0.05,
        relative_entropy_loss=0.05,
        wall_time=False,
        joint_kl_penalty=0.1,
        target_kl=0.1,
    ):
        super(PG, self).__init__()
        config = locals()
        # Remove 'self' and other unwanted items
        config.pop("self")
        self.joint_kl_penalty = joint_kl_penalty
        self.target_kl = target_kl
        self.config = config
        self.wall_time = wall_time
        self.load_from_checkpoint = load_from_checkpoint
        self.relative_entropy_loss = relative_entropy_loss
        if self.load_from_checkpoint is not None:
            self.load(self.load_from_checkpoint)
            return

        # Set up the params for Qmixed PPO
        self.importance_temperature = importance_schedule[0]
        self.max_importance_temperature = importance_schedule[0]
        self.min_importance_temperature = importance_schedule[1]
        self.importance_temperature_steps = importance_schedule[2]
        self.importance_step = 0
        self.importance_from_grad = importance_from_grad
        self.softmax_importance_scale = softmax_importance_scale
        self.on_policy_mixer = on_policy_mixer
        self.mixer_dim = mixer_dim

        # Set up the normal PPO params
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = discrete_action_dims
        self.batch_name_map = batch_name_map
        self.eval_mode = eval_mode
        self.mix_type = mix_type
        self.mixer = None
        self.name = name
        self.encoder = encoder
        self.action_clamp_type = action_clamp_type
        self.naive_imitation = naive_imitation
        self.ppo_clip = ppo_clip
        self.value_clip = value_clip
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.mini_batch_size = mini_batch_size
        self.advantage_type = advantage_type
        self.clip_grad = clip_grad
        self.device = device
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.n_epochs = n_epochs
        self.activation = activation
        self.norm_advantages = norm_advantages
        self.policy_loss = 1.0
        self.critic_loss_coef = value_loss_coef
        self.entropy_loss = entropy_loss
        self.min_actions = min_actions
        self.max_actions = max_actions
        self.hidden_dims = hidden_dims
        self.orthogonal = orthogonal
        self.std_type = std_type
        self.g_mean = 0
        self.steps = 0
        self.anneal_lr = anneal_lr
        self.lr = lr
        self.logit_reg = logit_reg
        self.mean_std = 1
        self.end_early = False

        self._sanitize_params()
        self._create_mixer()  # This needs to be before _get_torch_params for Adam to work
        self._get_torch_params(encoder, action_head_hidden_dims)

    def _sanitize_params(self):
        self.total_action_dims = 0
        if self.mix_type is not None and self.mix_type.lower() == "none":
            self.mix_type = None
        if (
            self.discrete_action_dims is not None
            and len(self.discrete_action_dims) == 0
        ):
            self.discrete_action_dims = None
        if self.mix_type is not None and self.mix_type.lower() == "none":
            self.mix_type = None
        for k in ["VDN", "QMIX"]:
            if self.mix_type is not None and self.mix_type.lower() == "vdn":
                self.mix_type = "VDN"
                self.advantage_type = "qmix"
            if self.mix_type is not None and self.mix_type.lower() == "qmix":
                self.mix_type = "QMIX"
                self.advantage_type = "qmix"

        if self.continuous_action_dim is not None and self.continuous_action_dim > 0:
            if isinstance(self.max_actions, list):
                self.max_actions = np.array(self.max_actions)
            if isinstance(self.min_actions, list):
                self.min_actions = np.array(self.min_actions)
            if isinstance(self.min_actions, np.ndarray):
                self.min_actions = torch.from_numpy(self.min_actions).to(self.device)
            if isinstance(self.max_actions, np.ndarray):
                self.max_actions = torch.from_numpy(self.max_actions).to(self.device)

        if self.discrete_action_dims is not None:
            self.total_action_dims += len(self.discrete_action_dims)
        if self.continuous_action_dim > 0:
            self.total_action_dims += self.continuous_action_dim

    def _assert_params(self):
        assert (
            self.continuous_action_dim > 0 or self.discrete_action_dims is not None
        ), "At least one action dim should be provided"
        for k in [
            "rewards",
            "obs",
            "obs_",
            "continuous_log_probs",
            "discrete_log_probs",
        ]:
            assert (
                k in self.batch_name_map
            ), "PPO needs these names defined ['rewards','obs','obs_'] "
        if self.discrete_action_dims is not None:
            assert (
                "discrete_actions" in self.batch_name_map
                and "discrete_log_probs" in self.batch_name_map
            ), 'discrete actions is not None but "discrete_actions" or "discrete_log_probs" does not appear in batch_name_map'
        if self.continuous_action_dim > 0:
            assert (
                "continuous_actions" in self.batch_name_map
                and "continuous_log_probs" in self.batch_name_map
            ), 'continuous actions is not None but "continuous_actions" or "continuous_log_probs" does not appear in batch_name_map'
        if self.continuous_action_dim > 0:
            assert (
                self.max_actions is not None or self.action_clamp_type is None
            ), "Clamp type is not None, but max actions is None so no way to clamp"
            assert (
                self.min_actions is not None or self.action_clamp_type is None
            ), "Clamp type is not None, but min actions is None so no way to clamp"

            if self.action_clamp_type is not None:
                assert (
                    self.max_actions is not None
                    and len(self.max_actions) >= self.continuous_action_dim
                ), f"If Clamp type '{self.action_clamp_type}' is not None, len(max_actions): {len(self.max_actions) if self.max_actions is not None else None}, must be greater than continuous_action_dim: {self.continuous_action_dim}"
                assert (
                    self.min_actions is not None
                    and len(self.min_actions) >= self.continuous_action_dim
                ), f"If Clamp type '{self.action_clamp_type}' is not None, len(min_actions): {len(self.min_actions) if self.min_actions is not None else None}, must be greater than continuous_action_dim: {self.continuous_action_dim}"
        if self.mix_type == "QMIX":
            assert self.mixer_dim is not None and isinstance(
                self.mixer_dim, int
            ), "mixer_dim must be an integer embedding size to use QMIX e.i. 256"
            assert (
                self.advantage_type == "qmix"
            ), "Cane have mixtype QMIX without advantage tyype qmix"
        assert self.advantage_type.lower() in [
            "gae",
            "a2c",
            "constant",
            "gv",
            "g",
            "qmix",  # one value and one A/U value per discrete head
        ], "Invalid advantage type"

    def _create_mixer(self):
        if self.mix_type is None:
            self.critic = ValueS(
                obs_dim=self.obs_dim,
                hidden_dim=self.hidden_dims[0],
                device=self.device,
                orthogonal_init=self.orthogonal,
                activation=self.activation,
            ).to(self.device)

        elif self.mix_type == "VDN":
            self.mixer = VDNMixer(
                self.total_action_dims, self.obs_dim, mixing_embed_dim=self.mixer_dim
            ).to(self.device)
            self.critic = QS(
                obs_dim=self.obs_dim,
                continuous_action_dim=self.continuous_action_dim,
                discrete_action_dims=self.discrete_action_dims,
                hidden_dims=[self.mixer_dim, self.mixer_dim],
                encoder=None,
                activation="tanh",
                dueling=True,
                device=self.device,
                n_c_action_bins=5,
                head_hidden_dims=[64],
                QMIX=False,
                QMIX_hidden_dim=0,
            ).to(self.device)
        elif self.mix_type == "QMIX":
            self.mixer = QMixer(
                self.total_action_dims, self.obs_dim, mixing_embed_dim=self.mixer_dim
            ).to(self.device)
            self.critic = QS(
                obs_dim=self.obs_dim,
                continuous_action_dim=self.continuous_action_dim,
                discrete_action_dims=self.discrete_action_dims,
                hidden_dims=[self.mixer_dim, self.mixer_dim],
                encoder=None,
                activation="tanh",
                dueling=True,
                device=self.device,
                n_c_action_bins=5,
                head_hidden_dims=[self.mixer_dim],
                QMIX=False,
                QMIX_hidden_dim=0,
            ).to(self.device)

    def _get_torch_params(self, encoder, action_head_hidden_dims=None):
        st = None
        if self.std_type in ["full", "diagonal"]:
            st = self.std_type
        np_maxes = None
        np_mins = None
        if isinstance(self.max_actions, torch.Tensor):
            np_maxes = self.max_actions.to("cpu").numpy()
        if isinstance(self.min_actions, torch.Tensor):
            np_mins = self.min_actions.to("cpu").numpy()
        self.actor = StochasticActor(
            obs_dim=self.obs_dim,
            continuous_action_dim=self.continuous_action_dim,
            discrete_action_dims=self.discrete_action_dims,
            max_actions=np_maxes,
            min_actions=np_mins,
            hidden_dims=self.hidden_dims,
            device=self.device,
            orthogonal_init=self.orthogonal,
            activation=self.activation,
            encoder=encoder,
            gumbel_tau=0,
            action_head_hidden_dims=action_head_hidden_dims,
            std_type=st,
            clamp_type=self.action_clamp_type,
            log_std_clamp_range=(-3.0, 1.0),
        ).to(self.device)
        self.log_std_clamp_range = (-3.0, 1.0)
        self.actor_logstd = None
        self.optimizer: torch.optim.Adam
        if self.std_type == "stateless":
            self.actor_logstd = nn.Parameter(
                torch.zeros(self.continuous_action_dim).to(self.device),
                requires_grad=True,
            )  # TODO: Check this for expand as
            self.actor_logstd.retain_grad()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, list):
            return np.stack(
                [
                    t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
                    for t in x
                ],
                axis=-1,
            )
        elif x is None:
            return None
        else:
            return np.array(x)

    # train_actions will take one or multiple actions if given a list of observations
    # this way the agent can be parameter shared in a batched fashion.
    def train_actions(self, observations, action_mask=None, step=False, debug=False):
        """
        Returns action dictionary of the form:
            {
                "discrete_actions": np.int[da1,da2,...],
                "continuous_actions": np.float[ca1,ca2,...],
                "discrete_log_probs": float(sum(dlp1,dlp2,...)),
                "continuous_log_probs": float(sum(clp1,clp2,...)),
                "act_time": t(seconds) if self.wall_time,
            }
        returns gradient free numpy arrays / floats. act_time is the wall clock time
        """
        t = 0
        if self.wall_time:
            t = time.time()
        if debug:
            print(f"  Testing PPO Train Actions: Observations: {observations}")
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device, dtype=torch.float)
        if action_mask is not None and not torch.is_tensor(action_mask):
            action_mask = torch.tensor(action_mask, dtype=torch.float).to(self.device)

        if debug:
            print(f"  After tensor check: Observations{observations}")
        # print(f"Observations: {observations.shape} {observations}")

        if step:
            self.steps += 1
        if self.anneal_lr > 0:
            frac = max(1.0 - (self.steps - 1.0) / self.anneal_lr, 0.0001)
            lrnow = frac * self.lr
            self.optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            continuous_logits, continuous_log_std_logits, discrete_action_logits = (
                self.actor(x=observations, action_mask=action_mask, debug=debug)
            )

            if continuous_log_std_logits is None and self.continuous_action_dim > 0:
                assert (
                    self.std_type == "stateless"
                ), "Log std logits should only be none if we don't want the actor producing them aka stateless"
                continuous_log_std_logits = self.actor_logstd
            if debug:
                print(
                    f"  After actor: clog {continuous_logits}, dlog{discrete_action_logits}"
                )

            try:
                (
                    discrete_actions,
                    continuous_actions,
                    discrete_log_probs,
                    continuous_log_probs,
                    raw_continuous_activations,
                ) = self.actor.action_from_logits(
                    continuous_logits,
                    continuous_log_std_logits,
                    discrete_action_logits,
                    False,
                    self.continuous_action_dim > 0,
                    self.discrete_action_dims is not None,
                )
            except Exception as e:
                if continuous_logits is not None:
                    print(f"clogit train actions: {continuous_logits}")
                    print(f"clogstd train actions: {continuous_log_std_logits}")
                if discrete_action_logits is not None:
                    print(f"dlogit train actions: {discrete_action_logits}")
                print(self.actor)
                print(self.actor.device)
                print(e)
                raise (e)
        # print("train actions: ")
        # print(continuous_logits)
        # print(continuous_actions)
        # print(f"in train action lp: {continuous_log_probs}")
        if self.wall_time:
            t = time.time() - t
        return {
            "discrete_actions": self._to_numpy(discrete_actions),
            "continuous_actions": self._to_numpy(continuous_actions),
            "discrete_log_probs": self._to_numpy(discrete_log_probs),
            "continuous_log_probs": self._to_numpy(continuous_log_probs),
            "act_time": t,
        }

    def stable_greedy(self, obs, legal_action):
        ad = self.train_actions(
            observations=obs, action_mask=legal_action, step=False, debug=False
        )
        adiscrete, acontinuous = None, None
        if ad["discrete_actions"] is not None:
            adiscrete = torch.tensor(ad["discrete_actions"]).to(self.device)
        if ad["continuous_actions"] is not None:
            acontinuous = torch.tensor(ad["continuous_actions"]).to(self.device)
        return adiscrete, acontinuous

    # takes the observations and returns the action with the highest probability
    def ego_actions(self, observations, action_mask=None):
        with torch.no_grad():
            continuous_logits, continuous_log_std_logits, discrete_action_logits = (
                self.actor(x=observations, action_mask=action_mask, debug=False)
            )
            # TODO: Make it so that action_from_logits has ego version
            (
                discrete_actions,
                continuous_actions,
                discrete_log_probs,
                continuous_log_probs,
                _,
            ) = self.actor.action_from_logits(
                continuous_logits,
                continuous_log_std_logits,
                discrete_action_logits,
                False,
                False,
                False,
            )
            return {
                "discrete_actions": self._to_numpy(discrete_actions),
                "continuous_actions": self._to_numpy(continuous_actions),
            }

    def _discrete_imitation_loss(self, discrete_logits, discrete_actions):
        """
        Calculates the total cross-entropy loss for multiple discrete action dimensions.
        Args:
            discrete_logits (list of torch.Tensor): A list where each element is the logits
                for an action dimension. `discrete_logits[i]` has shape
                [batch_size, num_categories_in_dim_i].

            discrete_actions (torch.Tensor): The expert actions, with shape
                [batch_size, num_action_dims].
        Returns:
            torch.Tensor: A single scalar value representing the sum of losses.
        """
        total_loss = torch.zeros(1, device=self.device)
        # Iterate through each action dimension
        for i, single_dimension_logits in enumerate(discrete_logits):
            # Get the target actions for the current dimension (i)
            target_actions_for_dim = discrete_actions[:, i]
            # Calculate the cross-entropy loss for this dimension
            loss_for_dim = F.cross_entropy(
                single_dimension_logits, target_actions_for_dim
            )
            total_loss += loss_for_dim

        return total_loss

    def _continuous_mle_imitation_loss(
        self, continuous_mean_logits, continuous_log_std_logits, continuous_actions
    ):
        # if self.std_type == 'stateless' then we have a single nn parameter
        # called actor_logstd which does not depend on the state or action dimension.
        # if self.std_type == 'diagonal' then there will be one std_dev per sample
        # so that the std is constant accross action dimensions but it is stateful
        # if self.std_type == 'full' then there will be one std per output dimension
        # per sample, so expand_as will do nothing
        # In this case we are going with out self.actorlogstd
        if continuous_log_std_logits is None or self.std_type == "stateless":
            continuous_log_std_logits = self.actor_logstd
        assert (
            continuous_log_std_logits is not None
        ), f"Inside _continuous_mle_imitation_loss: log std logits is none for type: {self.std_type}"

        continuous_log_std_logits.expand_as(continuous_mean_logits)

        # If self.action_clamp_type == tanh, then we will use tanh to clamp both the
        # action ranges and standard deviations of the output distribution.
        # Otherwise we always clamp the standard deviation at least
        # If self.action_clamp_type == 'clamp' then we will clamp our own output actions
        # but this doesnt effect the loss function
        if self.action_clamp_type == "tanh":
            continuous_log_std_logits = torch.tanh(continuous_log_std_logits)
            continuous_log_std_logits = self.actor.log_std_clamp_range[0] + 0.5 * (
                self.actor.log_std_clamp_range[1] - self.actor.log_std_clamp_range[0]
            ) * (continuous_log_std_logits + 1)
        else:
            continuous_log_std_logits = torch.clamp(
                continuous_log_std_logits,
                self.actor.log_std_clamp_range[0],
                self.actor.log_std_clamp_range[1],
            )

        dist = torch.distributions.Normal(
            loc=continuous_mean_logits, scale=torch.exp(continuous_log_std_logits)
        )
        if self.action_clamp_type == "tanh":
            dist = TransformedDistribution(dist, TanhTransform())
            continuous_actions = minmaxnorm(
                continuous_actions, self.min_actions, self.max_actions
            )

        loss = (
            -dist.log_prob(continuous_actions).sum(dim=-1).mean()
        )  # TODO: dist.entropy() to stop it from overfitting
        return loss

    def _continuous_naive_imitation_loss(
        self,
        continuous_mean_logits: torch.Tensor,
        continuous_log_std_logits: torch.Tensor,
        continuous_actions: torch.Tensor,
        std_target=0.1,
    ):
        """
        Calculates a naive imitation loss using Mean Squared Error (MSE).

        This loss is composed of two parts:
        1. MSE between the clamped/squashed predicted mean and the expert actions.
        2. MSE between the predicted standard deviation and a fixed target std (0.1).
        """
        # --- 1. Process and Calculate Loss for Standard Deviation ---

        # Handle different std_types ('stateless', 'diagonal', 'full')
        if continuous_log_std_logits is None or self.std_type == "stateless":
            assert self.actor_logstd is not None and isinstance(
                self.actor_logstd, torch.Tensor
            )
            continuous_log_std_logits = self.actor_logstd

        assert (
            continuous_log_std_logits is not None
        ), f"Inside _continuous_naive_imitation_loss: log std logits is none for type: {self.std_type}"

        continuous_log_std_logits = continuous_log_std_logits.expand_as(
            continuous_mean_logits
        )

        # Clamp or squash the log_std logits based on the clamp type
        if self.action_clamp_type == "tanh":
            continuous_log_std_logits = torch.tanh(continuous_log_std_logits)
            # Rescale from [-1, 1] to the defined clamp range
            continuous_log_std_logits = self.actor.log_std_clamp_range[0] + 0.5 * (
                self.actor.log_std_clamp_range[1] - self.actor.log_std_clamp_range[0]
            ) * (continuous_log_std_logits + 1)
        else:
            continuous_log_std_logits = torch.clamp(
                continuous_log_std_logits,
                self.actor.log_std_clamp_range[0],
                self.actor.log_std_clamp_range[1],
            )

        # Calculate the predicted standard deviation
        predicted_std = torch.exp(continuous_log_std_logits)

        # Create a target std tensor with the same shape and a fixed value (e.g., 0.1)
        target_std = torch.full_like(predicted_std, std_target)

        # Calculate the MSE loss for the standard deviation
        std_loss = F.mse_loss(predicted_std, target_std)

        # --- 2. Process and Calculate Loss for the Mean ---

        # Apply the appropriate transformation to the predicted mean before calculating loss
        if self.action_clamp_type == "tanh":
            # Squash raw logits to [-1, 1]
            processed_mean = torch.tanh(continuous_mean_logits)
            # Denormalize from [-1, 1] to the environment's action space [min, max]
            assert isinstance(self.min_actions, torch.Tensor)
            assert isinstance(self.max_actions, torch.Tensor)

            final_mean = self.min_actions + 0.5 * (
                self.max_actions - self.min_actions
            ) * (processed_mean + 1)

        elif self.action_clamp_type == "clamp":
            # Clamp the raw logits directly to the environment's action space
            assert (
                isinstance(self.min_actions, torch.Tensor)
                and isinstance(continuous_mean_logits, torch.Tensor)
                and isinstance(self.max_actions, torch.Tensor)
            )
            final_mean = torch.clamp(
                continuous_mean_logits, self.min_actions, self.max_actions
            )

        else:  # 'None'
            # Use the raw logits as the final mean
            final_mean = continuous_mean_logits

        # Calculate the MSE loss for the mean
        mean_loss = F.mse_loss(final_mean, continuous_actions)

        # --- 3. Combine Losses ---
        total_loss = mean_loss + std_loss

        return total_loss

    def imitation_learn(
        self,
        observations,
        continuous_actions=None,
        discrete_actions=None,
        action_mask=None,
        debug=False,
    ):
        t = 0
        if self.wall_time:
            t = time.time()
        continuous_mean_logits, continuous_log_std_logits, discrete_logits = self.actor(
            x=observations, action_mask=action_mask, debug=False
        )
        continuous_imitation_loss = torch.zeros(1, device=self.device)
        discrete_imitation_loss = torch.zeros(1, device=self.device)

        if self.continuous_action_dim > 0 and continuous_actions is not None:
            if self.naive_imitation:
                continuous_imitation_loss = self._continuous_mle_imitation_loss(
                    continuous_mean_logits,
                    continuous_log_std_logits,
                    continuous_actions,
                )
            else:
                continuous_imitation_loss = self._continuous_naive_imitation_loss(
                    continuous_mean_logits,
                    continuous_log_std_logits,
                    continuous_actions,
                    0.1,
                )
        if self.discrete_action_dims is not None and discrete_actions is not None:
            discrete_imitation_loss = self._discrete_imitation_loss(
                discrete_logits, discrete_actions
            )

        loss = discrete_imitation_loss + continuous_imitation_loss
        self.optimizer.zero_grad()
        loss.backward()  # type:ignore  started as a float
        self.optimizer.step()

        if isinstance(discrete_imitation_loss, torch.Tensor):
            discrete_imitation_loss = discrete_imitation_loss.to("cpu").item()
        if isinstance(continuous_imitation_loss, torch.Tensor):
            continuous_imitation_loss = continuous_imitation_loss.to("cpu").item()
        if self.wall_time:
            t = time.time() - t
        return {
            "im_discrete_loss": discrete_imitation_loss,
            "im_continuous_loss": continuous_imitation_loss,
            "im_time": t,
        }

    def utility_function(self, observations, actions=None):
        if not torch.is_tensor(observations):
            observations = torch.tensor(observations, dtype=torch.float).to(self.device)
        if actions is not None:
            return self.critic(observations, actions)
        else:
            return self.critic(observations)
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action=None):
        if self.mix_type is None:
            return self.critic(obs).squeeze(-1)
        else:
            values, disc_advantages, cont_advantages = self.critic(obs)
            return values.squeeze(-1)

    def _get_disc_log_probs_entropy(self, logits, actions):
        log_probs = torch.zeros_like(actions, dtype=torch.float)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        # print("Disc probs:", log_probs.mean(dim=0).detach().cpu().numpy())
        return log_probs, dist.entropy().mean()

    def _get_cont_log_probs_entropy(
        self, logits, actions, lstd_logits: torch.Tensor | None = None
    ):
        lstd = -1.0
        if self.actor_logstd is not None:
            lstd = self.actor_logstd.expand_as(logits)
        else:
            assert (
                lstd_logits is not None
            ), "If the actor doesnt generate logits then it needs to have a global logstd"
            lstd = lstd_logits.expand_as(logits)

        # print(lstd.mean(dim=0).detach().cpu().numpy())
        # print(actions.abs().mean(dim=0).detach().cpu().numpy())
        # print(self.std_type)

        if self.action_clamp_type == "tanh":
            dist = torch.distributions.Normal(
                loc=torch.clip(logits, -4.0, 4.0), scale=torch.exp(lstd)
            )
            # dist = TransformedDistribution(dist, TanhTransform())
            # print("actions were tanhed so we need to get form raw to dist activations")
            # print(f"actions: {actions[:,0]}")
            activations = minmaxnorm(actions, self.min_actions, self.max_actions)
            activations = torch.clamp(activations, -0.999329299739, 0.999329299739)
            activations = torch.atanh(activations)
            # print(f"inverse tanh actions: {activations[:,0]}")
            # print(
            #    f"from raw logit means: {logits} and scale {torch.clip(torch.exp(lstd), min=1e-6)}"
            # )

        else:
            dist = torch.distributions.Normal(loc=logits, scale=torch.exp(lstd))
            activations = actions

        log_probs = dist.log_prob(activations).sum(dim=-1)
        # correction = torch.zeros_like(log_probs, dtype=torch.float)

        if self.action_clamp_type == "tanh":
            # print(
            #    f"log prob shape: {log_probs.shape} activ shape: {activations.shape} softplus thing: {F.softplus(-2 * activations).shape}"
            # )
            log_probs -= 2 * (
                np.log(2) - activations - F.softplus(-2 * activations)
            ).sum(dim=-1)

        if torch.min(log_probs) < -100:
            self.end_early = True
            print(
                f"{self.action_clamp_type} Warning: log_probs has very low values: {torch.min(log_probs)}. "
                "This might cause numerical instability."
            )
            # print(log_probs < -20)
            # print(
            #     f"loc: {logits[log_probs < -20]}, scale: {torch.exp(lstd)[log_probs < -20]} activations: {activations[log_probs < -20]}"
            # )
            # print(
            #     f"diff: {(torch.clip(logits, -4.0, 4.0) - activations)[log_probs < -20]}"
            # )
            # print(f"lstd: {torch.exp(lstd)[log_probs < -20]}")
            # print(f"log probs: {log_probs[log_probs < -20]}")
            # eloss = 0.0
            # print(f"lstd: {torch.exp(lstd)[log_probs < -20]}")
            # input(f"dist entropy: {dist.entropy()[log_probs < -20]}")
        # else:
        # eloss = dist.entropy().mean()
        #     eloss = 0.0
        #     input(f"print the rest? {self.action_clamp_type}")
        #     print(
        #         f"loc: {logits}, scale: {torch.exp(lstd)} actions: {actions} activations {activations}"
        #     )
        # log_probs = torch.clamp(log_probs, -100, 2)
        # eloss = eloss * 100
        eloss = dist.entropy().mean()
        return log_probs, eloss

    def _print_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p is None or p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        print(total_norm)

    def _critic_loss(
        self, batch: FlexiBatch, indices, G, agent_num=0, debug=False
    ) -> torch.Tensor:
        V_current = self.critic(
            batch.__getattr__(self.batch_name_map["obs"])[agent_num, indices]
        )
        critic_loss = 0.5 * ((V_current - G[indices]) ** 2).mean()
        return critic_loss

    def _calculate_advantages(self, batch: FlexiBatch, agent_num=0, debug=False):
        assert isinstance(
            batch.terminated, torch.Tensor
        ), "need to send batch to torch first"

        values = None
        rewards = batch.__getattr__(self.batch_name_map["rewards"])
        last_val = self.expected_V(
            batch.__getattr__(self.batch_name_map["obs_"])[agent_num, -1], None
        )
        if self.advantage_type == "gv":
            G = FlexibleBuffer.G(
                rewards,
                batch.terminated,
                last_value=last_val.item(),
                gamma=self.gamma,
            )
            advantages = G - self.critic(
                batch.__getattr__(self.batch_name_map["obs"])[agent_num]
            )
        elif self.advantage_type == "constant":
            # print(
            #    f"constant shapes: {rewards.shape} term {batch.terminated.shape} lastval: {last_val}"
            # )
            G = FlexibleBuffer.G(
                rewards,
                batch.terminated,
                last_value=last_val.item(),
                gamma=self.gamma,
            )
            self.g_mean = 0.9 * self.g_mean + 0.1 * G.mean()
            advantages = G - self.g_mean
        elif self.advantage_type == "g":
            G = FlexibleBuffer.G(
                rewards,
                batch.terminated,
                last_value=last_val.item(),
                gamma=self.gamma,
            )
            advantages = G
        elif self.advantage_type in ["gae", "a2c"]:
            with torch.no_grad():
                if "values" in self.batch_name_map.keys():
                    values = batch.__getattr__(self.batch_name_map["values"])[agent_num]
                elif hasattr(batch, "values"):
                    values = batch.__getattr__("values")[agent_num]
                else:
                    values = self.critic(
                        batch.__getattr__(self.batch_name_map["obs"])[agent_num]
                    ).squeeze(-1)

            # values = values.squeeze(-1)
            if self.advantage_type == "gae":
                # print(
                #    f"rewards: {rewards.shape}, values: {values.shape} terminated: {batch.terminated.shape}"
                # )
                # input("hmm2")
                G, advantages = FlexibleBuffer.GAE(
                    rewards,
                    values,
                    batch.terminated,
                    last_val,
                    self.gamma,
                    self.gae_lambda,
                )
            elif self.advantage_type == "a2c":
                G, advantages = FlexibleBuffer.GAE(
                    rewards,
                    values,
                    batch.terminated,
                    last_val,
                    self.gamma,
                    0.0,
                )
            else:
                raise ValueError("Invalid advantage type")
        # elif self.advantage_type == "qmix":
        #     with torch.no_grad():
        #         if "values" in self.batch_name_map.keys():
        #             values = batch.__getattr__(self.batch_name_map["values"])[agent_num]
        #         elif hasattr(batch, "values"):
        #             values = batch.__getattr__("values")[agent_num]
        #         else:
        #             values, da, ca = self.critic(
        #                 batch.__getattr__(self.batch_name_map["obs"])[agent_num]
        #             ).squeeze(-1)

        #     G, advantages = FlexibleBuffer.GAE(
        #         rewards,
        #         values,
        #         batch.terminated,
        #         last_val,
        #         self.gamma,
        #         self.gae_lambda,
        #     )
        else:
            raise Exception(f"advantage type {self.advantage_type} not allowed")
        if debug:
            print(
                f"  batch rewards: {batch.__getattr__(self.batch_name_map['rewards'])}"
            )
            print(
                f"  raw critic: {self.critic(batch.__getattr__(self.batch_name_map['obs']))}"
            )
            print(f"  Advantages: {advantages}")
            print(f"  G: {G}")
        return G, advantages, values

    def _continuous_actor_loss(
        self, action_means, action_log_std, old_log_probs, advantages, actions
    ):
        if len(advantages.shape) > 1:
            advantages = advantages.squeeze(-1)
        cont_log_probs, cont_entropy = self._get_cont_log_probs_entropy(
            logits=action_means,
            actions=actions,
            lstd_logits=action_log_std,
        )

        if self.ppo_clip > 0:
            if cont_log_probs.ndim != old_log_probs.ndim:
                raise Exception(
                    f"Something is wrong: clp {cont_log_probs.shape} oldlp {old_log_probs.shape}"
                )
            logratio = cont_log_probs - old_log_probs
            # batch.continuous_log_probs[agent_num, indices]

            ratio = logratio.exp()
            pg_loss1 = advantages * ratio
            pg_loss2 = advantages * torch.clamp(
                ratio, 1 - self.ppo_clip, 1 + self.ppo_clip
            )

            continuous_policy_gradient = torch.min(pg_loss1, pg_loss2)
        else:
            continuous_policy_gradient = cont_log_probs * advantages
        actor_loss = (
            -self.policy_loss * continuous_policy_gradient.mean()
            - self.entropy_loss * cont_entropy
        )
        al = self.logit_reg * (action_means[torch.abs(action_means) > 4.0] ** 2).mean()
        if torch.isnan(actor_loss):
            actor_loss = 0.0
        if not torch.isnan(al):
            actor_loss += al
        self.result_dict["c_entropy"] += cont_entropy.item()
        return actor_loss

    def _discrete_actor_loss(self, actions, log_probs, logits, advantages):
        actor_loss = torch.zeros(1, device=self.device)
        for head in range(actions.shape[-1]):
            dist = Categorical(logits=logits[head])  # TODO: th
            entropy = dist.entropy().mean()
            selected_log_probs = dist.log_prob(actions[:, head])
            if self.ppo_clip > 0:
                old_lp = log_probs[:, head]
                assert (
                    old_lp.ndim == selected_log_probs.ndim
                ), f"Log prob dims differ: old: {old_lp.shape} new: {selected_log_probs.shape}"
                logratio = (
                    selected_log_probs
                    - old_lp  # batch.discrete_log_probs[agent_num, indices, head]
                )
                ratio = logratio.exp()
                pg_loss1 = advantages.squeeze(-1) * ratio
                pg_loss2 = advantages.squeeze(-1) * torch.clamp(
                    ratio, 1 - self.ppo_clip, 1 + self.ppo_clip
                )
                discrete_policy_gradient = torch.min(pg_loss1, pg_loss2)
            else:
                discrete_policy_gradient = selected_log_probs * advantages.squeeze(-1)

            actor_loss += (
                -self.policy_loss * discrete_policy_gradient.mean()
                - self.entropy_loss * entropy
            )
            self.result_dict["d_entropy"] += entropy.item()

        return actor_loss

    def reinforcement_learn(
        self,
        batch: FlexiBatch,
        agent_num=0,
        critic_only=False,
        debug=False,
    ):
        t = 0
        if self.wall_time:
            t = time.time()
        if self.eval_mode:
            return {
                "rl_actor_loss": 0,
                "rl_critic_loss": 0,
                "d_entropy": 0,
                "c_entropy": 0,
                "c_std": 0,
                "rl_time": 0,
            }
        self.result_dict = {
            "rl_actor_loss": 0,
            "rl_critic_loss": 0,
            "d_entropy": 0,
            "c_entropy": 0,
            "c_std": 0,
            "rl_time": t,
        }
        # print(f"mix type: {self.mix_type}, adv type: {self.advantage_type}")
        if self.mix_type == "QMIX" or self.mix_type == "VDN":
            return self._mix_reinforcement_learn(batch, agent_num, critic_only, debug)
        if debug:
            print(f"Starting PG Reinforcement Learn for agent {agent_num}")
        with torch.no_grad():
            G, advantages, values = self._calculate_advantages(batch, agent_num, debug)
        assert isinstance(
            advantages, torch.Tensor
        ), "Advantages has to be a tensor but it isn't, maybe batch was not called with as_torch=True?"
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        avg_actor_loss = 0
        avg_critic_loss = 0
        # Update the actor
        action_mask = None
        if batch.action_mask is not None:
            action_mask = batch.action_mask[agent_num]  # TODO: Unit test this later
            if action_mask is not None:
                print("Action mask Not implemented yet")

        assert isinstance(
            batch.terminated, torch.Tensor
        ), "need to send batch to torch first"
        bsize = len(batch.terminated)
        nbatch = bsize // self.mini_batch_size
        mini_batch_indices = np.arange(len(batch.terminated))
        np.random.shuffle(mini_batch_indices)

        if debug:
            print(
                f"  bsize: {bsize}, Mini batch indices: {mini_batch_indices}, nbatch: {nbatch}"
            )

        bnum = 0
        self.end_early = False
        for epoch in range(self.n_epochs):
            if self.end_early:
                bnum = max(0.01, bnum)
                break
            if debug:
                print("  Starting epoch", epoch)

            while self.mini_batch_size * bnum < bsize:
                # Get Critic Loss
                if self.end_early:
                    break
                bstart = self.mini_batch_size * bnum
                bend = min(bstart + self.mini_batch_size, bsize - 1)
                indices = mini_batch_indices[bstart:bend]
                bnum += 1
                # print(f"bnum: {bnum}")
                if debug:
                    print(
                        f"    Mini batch: {bstart}:{bend}, Indices: {indices}, {len(indices)}"
                    )

                critic_loss = self._critic_loss(batch, indices, G, agent_num, debug)
                actor_loss = torch.zeros(1, device=self.device)
                if not critic_only:
                    mb_adv = advantages[torch.from_numpy(indices).to(self.device)]
                    mb_obs = batch.__getattr__(self.batch_name_map["obs"])[
                        agent_num, indices
                    ]
                    continuous_means, continuous_log_std_logits, discrete_logits = (
                        self.actor(x=mb_obs)
                    )
                    if self.continuous_action_dim > 0:
                        clp = batch.__getattr__(
                            self.batch_name_map["continuous_log_probs"]
                        )[agent_num, indices]
                        cact = batch.__getattr__(
                            self.batch_name_map["continuous_actions"]
                        )[agent_num, indices]
                        actor_loss += self._continuous_actor_loss(
                            continuous_means,
                            continuous_log_std_logits,
                            clp,
                            mb_adv,
                            cact,
                        )
                    if self.discrete_action_dims is not None:
                        dact = batch.__getattr__(
                            self.batch_name_map["discrete_actions"]
                        )[agent_num, indices]
                        dlp = batch.__getattr__(
                            self.batch_name_map["discrete_log_probs"]
                        )[agent_num, indices]
                        actor_loss += self._discrete_actor_loss(
                            dact, dlp, discrete_logits, mb_adv
                        )

                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss * self.critic_loss_coef
                loss.backward()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        0.5,
                        error_if_nonfinite=True,
                        foreach=True,
                    )

                self.optimizer.step()

                avg_actor_loss += actor_loss.to("cpu").detach().item()
                avg_critic_loss += critic_loss.to("cpu").detach().item()
            # avg_actor_loss /= nbatch
            # avg_critic_loss /= nbatch
            # print(f"actor_loss: {actor_loss.item()}")

        avg_actor_loss /= bnum
        avg_critic_loss /= bnum
        if self.wall_time:
            t = time.time() - t
            self.result_dict["rl_time"] = t
        self.result_dict["rl_actor_loss"] = avg_actor_loss
        self.result_dict["rl_critic_loss"] = avg_critic_loss
        return self.result_dict

    def _bin_continuous_actions(self, c_actions):
        """Given continuous actions we return the discretized bins that the critic is using"""
        assert (
            self.min_actions is not None and self.max_actions is not None
        ), "Can't bin actions with no max and min action"
        n_bins = 5
        min_actions = self.min_actions.unsqueeze(0)  # type:ignore
        max_actions = self.max_actions.unsqueeze(0)  # type:ignore
        bin_width = (max_actions - min_actions) / (n_bins - 1)
        bin_indices = torch.round((c_actions - min_actions) / bin_width)
        bin_indices = bin_indices.clamp(0, n_bins - 1)
        return bin_indices.long()

    def _gather_observed_advantages(self, d_adv, c_adv, d_actions, c_actions):
        advantages = []
        if d_adv is not None:
            for h in range(len(d_adv)):
                adv_h = d_adv[h]
                assert isinstance(adv_h, torch.Tensor)
                advantages.append(
                    adv_h.gather(dim=-1, index=d_actions[:, h].unsqueeze(-1))
                )
        if c_adv is not None:
            c_indices = self._bin_continuous_actions(c_actions)
            assert isinstance(c_adv, torch.Tensor)
            advantages.append(
                c_adv.gather(dim=-1, index=c_indices.unsqueeze(-1)).squeeze(-1)
            )
        advantages = torch.cat(advantages, dim=-1)
        return advantages

    def _max_advantages(self, d_adv, c_adv):
        advantages = []
        if d_adv is not None:
            for h in range(len(d_adv)):
                adv_h = d_adv[h]
                assert isinstance(adv_h, torch.Tensor)
                advantages.append(adv_h.max(dim=-1).values.unsqueeze(-1))
        if c_adv is not None:
            assert isinstance(c_adv, torch.Tensor)
            advantages.append(c_adv.max(dim=-1).values)
        advantages = torch.cat(advantages, dim=-1)
        return advantages

    def _gather_importance(self, d_adv, c_adv):
        importance = []
        if d_adv is not None:
            for h in range(len(d_adv)):
                adv_h = d_adv[h].detach()
                assert isinstance(adv_h, torch.Tensor)
                importance.append(
                    adv_h.max(dim=-1).values.unsqueeze(-1)
                    - adv_h.min(dim=-1).values.unsqueeze(-1)
                )
        if c_adv is not None:
            assert isinstance(c_adv, torch.Tensor)
            importance.append(c_adv.max(dim=-1).values - c_adv.min(dim=-1).values)
        importance = torch.cat(importance, dim=-1)
        return importance

    def _update_importance(self):
        # based on importance step, max and min, and nsteps, scale importance temperature
        frac = min(self.importance_step / self.importance_temperature_steps, 1.0)
        self.importance_temperature = (
            self.max_importance_temperature * (1.0 - frac)
            + self.min_importance_temperature * frac
        )
        self.importance_step += 1

    def _weighted_gae(
        self,
        rewards: torch.Tensor,  # shape = [n_steps,1]
        values: torch.Tensor,  # shape = [n_steps]
        bootstrap_values: torch.Tensor,  # shape = [n_steps,1]
        terminated: torch.Tensor,  # shape = [n_steps]
        truncated: torch.Tensor,  # shape = [n_steps]
        advantage_weights: torch.Tensor,  # shape = [n_steps, n_agents]
        gamma=0.99,
        gae_lambda=0.95,
    ):
        advantages = torch.zeros_like(advantage_weights).to(advantage_weights.device)
        num_steps = len(rewards)
        last_gae_lam = torch.zeros(advantage_weights.shape[1]).to(self.device)
        for step in reversed(range(num_steps)):
            if terminated[step] > 0.1:
                next_value = 0.0
            else:
                next_value = gamma * bootstrap_values[step]

            ep_not_over = float((terminated[step] < 0.1) and (truncated[step] < 0.1))
            delta = rewards[step] + next_value - values[step]
            weighted_delta = delta * advantage_weights[step]
            last_gae_lam = (
                weighted_delta + gamma * gae_lambda * ep_not_over * last_gae_lam
            )
            # print(f"Step {step}: reward {rewards[step]}, next_value {next_value}, value {values[step]}, delta {delta}")
            # print(f"          advantage weight: {advantage_weights[step]}, weighted delta: {weighted_delta}, last_gae_lam: {last_gae_lam}")
            # if step>num_steps-5:
            #     input(f"step {step} done? ep_over: {ep_not_over}")
            advantages[step] = last_gae_lam
        G = advantages + values.unsqueeze(-1)
        # print(f"Weighted GAE advantages: {advantages}")
        # print(f"Weighted GAE G: {G}")
        # print(f"Weighted GAE values: {values}")
        # print(f"Rewards: {rewards}")
        # input()
        return G, advantages

    def _continuous_log_probs_per_dim(self, logits, lstd_logits, actions):
        lstd = -1.0
        if self.actor_logstd is not None:
            lstd = self.actor_logstd.expand_as(logits)
        else:
            assert (
                lstd_logits is not None
            ), "If the actor doesnt generate logits then it needs to have a global logstd"
            lstd = lstd_logits.expand_as(logits)
        # TODO: Make this track better, this is a hack
        self.mean_std = torch.exp(lstd.detach().mean(0).cpu())
        dist = torch.distributions.Normal(
            loc=torch.clip(logits, min=-4.0, max=4.0), scale=torch.exp(lstd)
        )
        if self.action_clamp_type == "tanh":
            activations = minmaxnorm(actions, self.min_actions, self.max_actions)
            activations = torch.clamp(activations, -0.999329299739, 0.999329299739)
            activations = torch.atanh(activations)
        else:
            activations = actions
        log_probs = dist.log_prob(activations)
        if self.action_clamp_type == "tanh":
            log_probs -= 2 * (np.log(2) - activations - F.softplus(-2 * activations))

        c_entropy = dist.entropy().sum(-1)

        if torch.min(log_probs) < -20:
            print(
                f"{self.action_clamp_type} Warning: log_probs has very low values: {torch.min(log_probs)}. "
                "This might cause numerical instability. a"
            )
        return log_probs, c_entropy, dist

    def _log_probs_per_dim(self, obs, d_actions, c_actions):
        continuous_means, continuous_log_std_logits, discrete_logits = self.actor(obs)

        # print(
        #     f"continuous_means: {continuous_means}\nclstdl: {continuous_log_std_logits} {self.std_type}"
        # )
        discrete_log_probs = None
        continuous_log_probs = None
        c_dist = None
        d_dist = None
        lp = []
        c_entropy = 0
        if self.continuous_action_dim > 0:
            continuous_log_probs, c_entropy, c_dist = (
                self._continuous_log_probs_per_dim(
                    continuous_means, continuous_log_std_logits, c_actions
                )
            )
            lp.append(continuous_log_probs)
            # print(f"continuous lp: {continuous_log_probs}")
        d_entropy = 0
        d_dists = None
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            d_dists = []
            discrete_log_probs = torch.zeros(
                discrete_logits[0].shape[0],
                len(self.discrete_action_dims),
                device=self.device,
            )
            for i, logits in enumerate(discrete_logits):
                d_dist = torch.distributions.Categorical(logits=logits)
                d_dists.append(d_dist)
                discrete_log_probs[:, i] = d_dist.log_prob(d_actions[:, i])
                d_entropy += d_dist.entropy().squeeze(-1)
            lp.append(discrete_log_probs)
        lp = torch.cat(lp, dim=-1)
        logit_regularization_loss = 0
        if self.continuous_action_dim > 0:
            logit_regularization_loss = (
                continuous_means[torch.abs(continuous_means) > 4.0] ** 2
            ).mean()
            # std_logit_reg = 0.0
            # if continuous_log_std_logits is not None:
            #     std_logit_reg = (
            #         continuous_log_std_logits[
            #             torch.abs(continuous_log_std_logits)
            #             > self.log_std_clamp_range[1] + 1e-3
            #         ]
            #         ** 2
            #     ).mean()
            #     std_logit_reg += (
            #         continuous_log_std_logits[
            #             torch.abs(continuous_log_std_logits)
            #             < self.log_std_clamp_range[0] - 1e-3
            #         ]
            #         ** 2
            #     ).mean()
            #     if not torch.isnan(std_logit_reg):
            #         logit_regularization_loss += std_logit_reg
            if torch.isnan(logit_regularization_loss):
                logit_regularization_loss = 0.0
        return lp, d_entropy, c_entropy, logit_regularization_loss, d_dists, c_dist

    def _mix_actor_loss(
        self,
        old_log_probs,
        new_log_probs,
        advantages,
        entropy,
        old_d_dists,
        new_d_dists,
        old_c_dists,
        new_c_dists,
    ):
        logratio = new_log_probs - old_log_probs
        assert (
            new_log_probs.ndim == old_log_probs.ndim
        ), f"new lp {new_log_probs.shape} old lp {old_log_probs.shape}"
        ratio = torch.exp(logratio)
        # ratio_joint = torch.exp(new_log_probs.sum(-1) - old_log_probs.sum(-1))
        clip_ratio = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        # print(f" log probs: new {new_log_probs}, old {old_log_probs}")
        # PG loss
        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * clip_ratio
        # print(pg_loss1.shape, pg_loss2.shape)
        policy_loss = -(torch.min(pg_loss1, pg_loss2).sum(-1)).mean()
        # print(f"policy loss before entropy and kl: {policy_loss.item()}")
        # print(f"entropy loss: {self.entropy_loss *entropy.mean().item()}")
        # Entropy Bonus
        policy_loss -= self.entropy_loss * entropy.mean()

        # KL Divergence for joint distribution
        kl_div = 0
        if old_d_dists is not None and new_d_dists is not None:
            for old_d, new_d in zip(old_d_dists, new_d_dists):
                # print("Discrete KL")
                # print(old_d)
                # print(new_d)
                kl_div += torch.distributions.kl_divergence(old_d, new_d)
                # print(kl_div.mean())
        if old_c_dists is not None and new_c_dists is not None:
            # print("Continuous KL")
            # print(old_c_dists)
            # print(new_c_dists)
            kl_div += torch.distributions.kl_divergence(old_c_dists, new_c_dists).sum(
                -1
            )
            # print(kl_div.mean())

        joint_kl = kl_div.mean()
        # print(f"KL Divergence loss: {self.joint_kl_penalty * joint_kl.item()}")
        policy_loss += self.joint_kl_penalty * joint_kl

        # print(f"joint_kl: {joint_kl.item()}, penalty: {self.joint_kl_penalty}")
        # print(f"policy_loss: {policy_loss}")
        if joint_kl > self.target_kl:
            self.joint_kl_penalty *= 1.5
        elif joint_kl < self.target_kl / 1.5:
            self.joint_kl_penalty /= 1.5
        self.joint_kl_penalty = min(max(self.joint_kl_penalty, 1e-4), 100)
        # input()

        return policy_loss, joint_kl

    def _mix_critic_only(self, batch: FlexiBatch, agent_num):
        obs = batch.__getattr__(self.batch_name_map["obs"])[agent_num]
        obs_ = batch.__getattr__(self.batch_name_map["obs_"])[agent_num]
        d_actions = batch.__getattr__(self.batch_name_map["discrete_actions"])[
            agent_num
        ]
        c_actions = batch.__getattr__(self.batch_name_map["continuous_actions"])[
            agent_num
        ]
        rewards = batch.__getattr__(self.batch_name_map["rewards"])
        terminated = batch.terminated
        truncated = batch.truncated
        if truncated is None:
            truncated = torch.zeros_like(rewards)
        values, d_adv, c_adv = self.critic(obs)
        adv = self._gather_observed_advantages(d_adv, c_adv, d_actions, c_actions)
        Q = (self.mixer(adv, obs)[0] + values).squeeze(-1)  # type:ignore

        with torch.no_grad():
            next_values, next_d_adv, next_c_adv = self.critic(obs_)
            if self.on_policy_mixer:
                next_adv = 0
                next_Q = next_values
            else:
                next_adv = self._max_advantages(
                    next_d_adv,
                    next_c_adv,
                )
                # print(
                #     f"self.mixer(next_adv, obs_)[0]: {self.mixer(next_adv, obs_)[0].shape}, next_values : {next_values.shape}"
                # )
                next_Q = (
                    self.mixer(next_adv, obs_)[0] + next_values  # type:ignore
                ).squeeze(-1)
                # print(f"next_Q: {next_Q.shape}")
            global_Q, global_GAE = self._weighted_gae(
                rewards=rewards,
                values=Q,
                bootstrap_values=next_Q,
                terminated=terminated,  # type:ignore
                truncated=truncated,  # type:ignore
                advantage_weights=torch.ones_like(rewards).unsqueeze(-1),
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        # print(f"Q - global_Q.squeeze(-1): {Q.shape} - {global_Q.squeeze(-1).shape}:")
        critic_loss = ((Q - global_Q.squeeze(-1)) ** 2).mean()
        self.optimizer.zero_grad()
        critic_loss.backward()
        # print(f"critic loss only: {critic_loss.item()}")
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            if self.mixer is not None and self.mix_type == "QMIX":
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 0.5)
        self.optimizer.step()

        return {
            "rl_critic_loss": critic_loss.item(),
        }

    def _get_dists(self, obs):
        d_dists, c_dist = None, None
        continuous_means, continuous_log_std_logits, discrete_logits = self.actor(obs)
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            d_dists = []
            for i, logits in enumerate(discrete_logits):
                d_dist = torch.distributions.Categorical(logits=logits)
                d_dists.append(d_dist)
        if self.continuous_action_dim > 0:
            lstd = -1.0
            if self.actor_logstd is not None:
                lstd = self.actor_logstd.expand_as(continuous_means)
            else:
                assert (
                    continuous_log_std_logits is not None
                ), "If the actor doesnt generate logits then it needs to have a global logstd"
                lstd = continuous_log_std_logits.expand_as(continuous_means)
            c_dist = torch.distributions.Normal(
                loc=torch.clip(continuous_means, min=-4.0, max=4.0),
                scale=torch.exp(lstd),
            )
        return d_dists, c_dist

    def _mix_reinforcement_learn(
        self, batch: FlexiBatch, agent_num, critic_only, debug
    ):
        """If we have QMIX going on then we need to do everything different so might as well make a new function"""
        assert self.mixer is not None, "Can't mix rl without a mixer..."
        t = 0
        if self.wall_time:
            t = time.time()
        if critic_only:
            return self._mix_critic_only(batch, agent_num)
        obs = batch.__getattr__(self.batch_name_map["obs"])[agent_num]
        obs_ = batch.__getattr__(self.batch_name_map["obs_"])[agent_num]
        d_actions = batch.__getattr__(self.batch_name_map["discrete_actions"])[
            agent_num
        ]
        c_actions = batch.__getattr__(self.batch_name_map["continuous_actions"])[
            agent_num
        ]
        rewards = batch.__getattr__(self.batch_name_map["rewards"])
        terminated = batch.terminated
        truncated = batch.truncated
        if truncated is None:
            truncated = torch.zeros_like(rewards)

        with torch.no_grad():
            values, d_adv, c_adv = self.critic(obs)
            adv = self._gather_observed_advantages(d_adv, c_adv, d_actions, c_actions)
        grad_free_adv = adv.detach()
        grad_free_adv.requires_grad = True
        __q, adv_grad = self.mixer(grad_free_adv, obs, with_grad=True)
        self.mixer.zero_grad()

        with torch.no_grad():
            Q = (self.mixer(adv, obs)[0] + values).squeeze(-1)
            next_values, next_d_adv, next_c_adv = self.critic(obs_)
            old_log_probs, old_d_ent, old_c_end, _, old_d_dist, old_c_dist = (
                self._log_probs_per_dim(obs, d_actions, c_actions)
            )
            if self.on_policy_mixer:
                next_adv = 0
                next_Q = next_values
            else:
                next_adv = self._max_advantages(
                    next_d_adv,
                    next_c_adv,
                )
                next_Q = (self.mixer(next_adv, obs_)[0] + next_values).squeeze(-1)
            # critic_target = rewards + self.gamma * (1.0 - terminated) * next_Q
            if self.importance_from_grad:
                # print(f" grad_free_adv: {grad_free_adv.shape}, adv_grad: {adv_grad.shape}")
                scaled_importance = grad_free_adv * adv_grad
            else:
                # print(f" raw_importance: {raw_importance.shape}, adv_grad: {adv_grad.shape}")
                raw_importance = self._gather_importance(d_adv, c_adv.detach())
                scaled_importance = raw_importance * adv_grad
            # print("raw importance:")
            # print(scaled_importance)
            # scaled_importance = scaled_importance*0 + 1.0 # testing
            # print("scaled importance before update:")
            # print(scaled_importance)
            self._update_importance()

            if self.softmax_importance_scale:
                scaled_importance = torch.softmax(
                    scaled_importance / self.importance_temperature, dim=-1
                )
                # print(scaled_importance)
            else:
                scaled_importance = (
                    scaled_importance.abs() + self.importance_temperature
                )
                scaled_importance /= scaled_importance.sum(dim=-1).unsqueeze(-1)

            # So learning rate doesn't shrink with number of agents
            # scaled_importance *= grad_free_adv.shape[-1]
            # print("scaled importance after update:")
            # print(scaled_importance)
            # input()
            G, gae = self._weighted_gae(
                rewards=rewards,
                values=Q,
                bootstrap_values=next_Q,
                terminated=terminated,  # type:ignore
                truncated=truncated,  # type:ignore
                advantage_weights=scaled_importance,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            # print(f"Q shape: {Q.shape}, G shape: {next_Q.shape}")
            global_Q, global_GAE = self._weighted_gae(
                rewards=rewards,
                values=Q,
                bootstrap_values=next_Q,
                terminated=terminated,  # type:ignore
                truncated=truncated,  # type:ignore
                advantage_weights=torch.ones_like(rewards).unsqueeze(-1),
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            # print(f"global_Q shape: {global_Q.shape}, global_G shape: {global_Q} gamma: {self.gamma}")
        # print(rewards)
        indices = np.arange(0, obs.shape[0])
        avg_actor_loss = 0.0
        avg_critic_loss = 0.0
        avg_d_entropy = 0.0
        avg_c_entropy = 0.0
        tot_joint_kl = 0.0
        bnum = 0

        # Getting old distributions for KL penalty

        gae = gae.detach()
        if self.norm_advantages:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        for k in range(self.n_epochs):
            # Shuffle indices at the start of each epoch
            np.random.shuffle(indices)

            with torch.no_grad():
                old_d_dists = []
                old_c_dists = []
                for start in range(0, len(indices), self.mini_batch_size):
                    end = start + self.mini_batch_size
                    mini_batch_indices = indices[start:end]
                    mb_obs = obs[mini_batch_indices]
                    # print(f"mb_obs: {mb_obs.shape}")
                    old_d_dist, old_c_dist = self._get_dists(mb_obs)
                    old_d_dists.append(old_d_dist)
                    old_c_dists.append(old_c_dist)
                # print(f"Got {len(old_d_dists)} old discrete dists and {len(old_c_dists)} old continuous dists")
                # print("old d dist example:")
                # if old_d_dists[0] is not None and len(old_d_dists[0]) > 0:
                #    print(old_d_dists[0][0].logits)
                # print("old c dist example:")
                # if old_c_dists[0] is not None:
                #    print(old_c_dists[0].loc, old_c_dists[0].scale)

            # TODO: Loop through mini-batches
            dist_steps = 0
            for start in range(0, len(indices), self.mini_batch_size):
                bnum += 1
                # Get mini batch indices
                end = start + self.mini_batch_size
                mini_batch_indices = indices[start:end]

                # Select mini-batch data
                mb_obs = obs[mini_batch_indices]
                mb_d_actions = d_actions[mini_batch_indices]
                mb_c_actions = c_actions[mini_batch_indices]
                mb_old_log_probs = old_log_probs[mini_batch_indices]
                mb_gae = gae[mini_batch_indices]
                mb_global_Q = global_Q[mini_batch_indices]

                # for m in range(10):
                #     mb_values, mb_d_adv, mb_c_adv = self.critic(mb_obs)
                #     mb_adv = self._gather_observed_advantages(
                #         mb_d_adv, mb_c_adv, mb_d_actions, mb_c_actions
                #     )
                #     mb_Q = (self.mixer(mb_adv, mb_obs)[0] + mb_values).squeeze(-1)
                #     if isinstance(mb_values, torch.Tensor):
                #         assert self.mixer(mb_adv, mb_obs)[0].shape == mb_values.shape, f"Shapes don't match in mix rl: {self.mixer(mb_adv, mb_obs)[0].shape} vs {mb_values.shape}"
                #         assert mb_Q.shape == mb_global_Q.squeeze(-1).shape, f"Shapes don't match in mix rl: {mb_Q.shape} vs {mb_global_Q.squeeze(-1).shape}"
                #     critic_loss = ((mb_Q - mb_global_Q.squeeze(-1)) ** 2).mean()
                #     self.optimizer.zero_grad()
                #     critic_loss.backward()
                #     self.optimizer.step()
                #     print(f"  Critic update {m}, loss: {critic_loss.item()}")
                mb_values, mb_d_adv, mb_c_adv = self.critic(mb_obs)
                mb_adv = self._gather_observed_advantages(
                    mb_d_adv, mb_c_adv, mb_d_actions, mb_c_actions
                )
                mb_Q = (self.mixer(mb_adv, mb_obs)[0] + mb_values).squeeze(-1)
                if isinstance(mb_values, torch.Tensor):
                    if isinstance(mb_values, torch.Tensor):
                        assert (
                            self.mixer(mb_adv, mb_obs)[0].shape == mb_values.shape
                        ), f"Shapes don't match in mix rl: {self.mixer(mb_adv, mb_obs)[0].shape} vs {mb_values.shape}"
                        assert (
                            mb_Q.shape == mb_global_Q.squeeze(-1).shape
                        ), f"Shapes don't match in mix rl: {mb_Q.shape} vs {mb_global_Q.squeeze(-1).shape}"
                critic_loss = ((mb_Q - mb_global_Q.squeeze(-1)) ** 2).mean()
                # input()
                # print(f"mb_Q: {mb_Q.mean()}, mb_global_Q: {mb_global_Q.squeeze(-1).mean()}")
                # input(f"are these similar? {(mb_Q - mb_global_Q.squeeze(-1)).mean()}")
                assert (
                    mb_Q.shape == mb_global_Q.squeeze(-1).shape
                ), f"Shapes don't match in mix rl: {mb_Q.shape} vs {mb_global_Q.squeeze(-1).shape}"
                # print(
                #     f" mb_Q: {mb_Q.shape}, mb_global_Q: {mb_global_Q.squeeze(-1).shape}"
                # )
                # print(f"critic loss: {critic_loss.item()}")

                (
                    mb_new_log_probs,
                    mb_d_entropy,
                    mb_c_entropy,
                    mb_logit_regulrization,
                    mb_d_dist,
                    mb_c_dist,
                ) = self._log_probs_per_dim(mb_obs, mb_d_actions, mb_c_actions)
                # print(mb_d_entropy, mb_c_entropy)
                if isinstance(mb_d_entropy, torch.Tensor) and isinstance(
                    mb_c_entropy, torch.Tensor
                ):
                    assert (
                        mb_d_entropy.shape == mb_c_entropy.shape
                    ), f"Entropy shapes don't match: {mb_d_entropy.shape} vs {mb_c_entropy.shape}"
                mb_entropy = mb_d_entropy + mb_c_entropy  # self.relative_entropy_loss *
                if torch.min(mb_new_log_probs) < -20:
                    print(
                        f"Bad log prob bnum {bnum} default to regularization only {torch.min(mb_new_log_probs)}"
                    )
                    # self.optimizer.state = collections.defaultdict(dict)
                    # continuous_means, continuous_log_std_logits, discrete_logits = (
                    #     self.actor(obs)
                    # )
                    # if self.continuous_action_dim > 0:
                    #     mask = continuous_means.abs() > 3.0
                    #     logit_regulrization = (
                    #         0.1 * ((mask * continuous_means) ** 2).mean()
                    #     )
                    #     self.optimizer.zero_grad()
                    #     logit_regulrization.backward()
                    #     self.optimizer.step()
                    # else:
                    #     logit_regulrization = 0
                    # continue

                # Sum and Normalize loss grads
                actor_loss, joint_kl = self._mix_actor_loss(
                    mb_old_log_probs,
                    mb_new_log_probs,
                    mb_gae,
                    mb_entropy,
                    old_d_dists[dist_steps],
                    mb_d_dist,
                    old_c_dists[dist_steps],
                    mb_c_dist,
                )
                actor_loss += mb_logit_regulrization

                loss = self.value_loss_coef * critic_loss + actor_loss
                self.optimizer.zero_grad()
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    if self.mixer is not None and self.mix_type == "QMIX":
                        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 0.5)
                loss.backward()
                self.optimizer.step()

                avg_actor_loss += actor_loss.item()
                avg_critic_loss += critic_loss.item()
                if isinstance(mb_c_entropy, torch.Tensor):
                    mb_c_entropy = mb_c_entropy.mean().cpu().item()
                avg_c_entropy += mb_c_entropy
                if isinstance(mb_d_entropy, torch.Tensor):
                    mb_d_entropy = mb_d_entropy.mean().cpu().item()
                avg_d_entropy += mb_d_entropy
                dist_steps += 1
                if isinstance(joint_kl, torch.Tensor):
                    tot_joint_kl += joint_kl.item()
        num_updates = bnum  # self.n_epochs * (len(indices) / self.mini_batch_size)
        avg_joint_kl = tot_joint_kl / num_updates
        avg_actor_loss /= num_updates
        avg_critic_loss /= num_updates
        avg_c_entropy /= num_updates
        avg_d_entropy /= num_updates
        if self.wall_time:
            t = time.time() - t

        result = {
            "rl_actor_loss": avg_actor_loss,
            "rl_critic_loss": avg_critic_loss,
            "d_entropy": avg_d_entropy,
            "c_entropy": avg_c_entropy,
            "c_std": self.mean_std,
            "rl_time": t,
            "joint_kl": avg_joint_kl,
        }
        # Turn any tensor params into numpy.
        # Detach gradient if has grad and send to cpu if on gpu
        for k in result.keys():
            v = result[k]
            if isinstance(v, torch.Tensor):
                v = v.detach()
                if v.requires_grad:
                    v = v.cpu()
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.cpu().numpy()
                result[k] = v
        return

    def _dump_attr(self, attr, path):
        f = open(path, "wb")
        pickle.dump(attr, f)
        f.close()

    def _load_attr(self, path):
        f = open(path, "rb")
        d = pickle.load(f)
        f.close()
        return d

    def save(self, checkpoint_path):
        if self.eval_mode:
            print("Not saving because model in eval mode")
            return
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(self.actor.state_dict(), checkpoint_path + "/PI")
        torch.save(self.critic.state_dict(), checkpoint_path + "/V")
        torch.save(self.actor_logstd, checkpoint_path + "/actor_logstd")
        # for i in range(len(self.attrs)):
        #    self._dump_attr(
        #        self.__dict__[self.attrs[i]], checkpoint_path + f"/{self.attrs[i]}"
        #    )

    def load(self, checkpoint_path):
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"

        # for i in range(len(self.attrs)):
        #    self.__dict__[self.attrs[i]] = self._load_attr(
        #        checkpoint_path + f"/{self.attrs[i]}"
        #    )
        self._get_torch_params(self.starting_actorlogstd)
        self.policy_loss = 5.0
        self.actor.load_state_dict(torch.load(checkpoint_path + "/PI"))
        self.critic.load_state_dict(torch.load(checkpoint_path + "/V"))
        self.actor_logstd = torch.load(checkpoint_path + "/actor_logstd")

    def __str__(self):
        st = ""
        for d in self.__dict__.keys():
            st += f"{d}: {self.__dict__[d]}\n"
        return st

    def param_count(self) -> tuple[int, int]:
        return super().param_count()
