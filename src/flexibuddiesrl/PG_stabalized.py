from .Agent import ValueS, StochasticActor, Agent
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
from typing import Any, cast


class PG(nn.Module, Agent):
    def __init__(
        self,
        obs_dim=10,
        continuous_action_dim=0,
        max_actions=None,
        min_actions=None,
        discrete_action_dims=None,
        lr=2.5e-3,
        gamma=0.99,
        n_epochs=2,
        device="cpu",
        entropy_loss=0.05,
        hidden_dims=[256, 256],
        activation="relu",
        ppo_clip=0.2,
        value_loss_coef=0.5,
        value_clip=0.5,
        advantage_type="gae",
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
        naive_immitation=False,  # if true, do MSE instead of MLE
        action_clamp_type="tanh",
        batch_name_map={
            "discrete_actions": "discrete_actions",
            "continuous_actions": "continuous_actions",
            "rewards": "global_rewards",
            "obs": "obs",
            "obs_": "obs_",
            "continuous_log_probs": "continuous_log_probs",
            "discrete_log_probs": "discrete_log_probs",
        },
    ):
        super(PG, self).__init__()
        self.eval_mode = eval_mode
        self.attrs = [
            "obs_dim",
            "continuous_action_dim",
            "max_actions",
            "min_actions",
            "discrete_action_dims",
            "lr",
            "gamma",
            "n_epochs",
            "device",
            "entropy_loss",
            "hidden_dims",
            "activation",
            "ppo_clip",
            "value_loss_coef",
            "value_clip",
            "advantage_type",
            "norm_advantages",
            "mini_batch_size",
            "anneal_lr",
            "orthogonal",
            "clip_grad",
            "gae_lambda",
            "g_mean",
            "steps",
            "eval_mode",
            "action_head_hidden_dims",
            "std_type",
            "naive_immitation",
            "action_clamp_type",
        ]
        self.run_times = {
            "advantage": 0.0,
            "critic_loss": 0.0,
            "act": 0.0,
            "dloss": 0.0,
            "closs": 0.0,
            "backward": 0.0,
            "tot": 0.0,
        }
        assert (
            continuous_action_dim > 0 or discrete_action_dims is not None
        ), "At least one action dim should be provided"

        self.batch_name_map = batch_name_map
        for k in ["rewards", "obs", "obs_"]:
            assert (
                k in batch_name_map
            ), "PPO needs these names defined ['rewards','obs','obs_'] "
        if discrete_action_dims is not None:
            assert (
                "discrete_actions" in batch_name_map
                and "discrete_log_probs" in batch_name_map
            ), 'discrete actions is not None but "discrete_actions" or "discrete_log_probs" does not appear in batch_name_map'
        if continuous_action_dim > 0:
            assert (
                "continuous_actions" in batch_name_map
                and "continuous_log_probs" in batch_name_map
            ), 'continuous actions is not None but "continuous_actions" or "continuous_log_probs" does not appear in batch_name_map'
        self.name = name
        self.encoder = encoder
        self.action_clamp_type = action_clamp_type
        self.naive_immitation = naive_immitation
        if load_from_checkpoint is not None:
            self.load(load_from_checkpoint)
            return
        self.ppo_clip = ppo_clip
        self.value_clip = value_clip
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.mini_batch_size = mini_batch_size
        assert advantage_type.lower() in [
            "gae",
            "a2c",
            "constant",
            "gv",
            "g",
        ], "Invalid advantage type"
        self.advantage_type = advantage_type
        self.clip_grad = clip_grad
        self.device = device
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = discrete_action_dims
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

        self._get_torch_params(encoder, action_head_hidden_dims)

        if self.continuous_action_dim is not None and self.continuous_action_dim > 0:
            if isinstance(self.max_actions, list):
                self.max_actions = np.array(self.max_actions)
            if isinstance(self.min_actions, list):
                self.min_actions = np.array(self.min_actions)

            if isinstance(self.min_actions, np.ndarray):
                self.min_actions = torch.from_numpy(min_actions).to(self.device)
            if isinstance(self.max_actions, np.ndarray):
                self.max_actions = torch.from_numpy(max_actions).to(self.device)

    def _get_torch_params(self, encoder, action_head_hidden_dims=None):
        st = None
        if self.std_type in ["full", "diagonal"]:
            st = self.std_type
        self.actor = StochasticActor(
            obs_dim=self.obs_dim,
            continuous_action_dim=self.continuous_action_dim,
            discrete_action_dims=self.discrete_action_dims,
            max_actions=self.max_actions,
            min_actions=self.min_actions,
            hidden_dims=self.hidden_dims,
            device=self.device,
            orthogonal_init=self.orthogonal,
            activation=self.activation,
            encoder=encoder,
            gumbel_tau=0,
            action_head_hidden_dims=action_head_hidden_dims,
            std_type=st,
        ).to(self.device)

        self.critic = ValueS(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dims[0],
            device=self.device,
            orthogonal_init=self.orthogonal,
            activation=self.activation,
        ).to(self.device)
        self.actor_logstd = None
        self.optimizer: torch.optim.Adam
        if self.std_type == "stateless":
            self.actor_logstd = nn.Parameter(
                torch.zeros(self.continuous_action_dim), requires_grad=True
            ).to(
                self.device
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
            # print(continuous_log_std_logits)
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
        return (
            self._to_numpy(discrete_actions),
            self._to_numpy(continuous_actions),
            self._to_numpy(discrete_log_probs),
            self._to_numpy(continuous_log_probs),
            0,  # vals.detach().cpu().numpy(), TODO: re-enable this when flexibuff is done
        )

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
            ) = self.actor.action_from_logits(
                continuous_logits,
                continuous_log_std_logits,
                discrete_action_logits,
                False,
                False,
                False,
            )
            return self._to_numpy(discrete_actions), self._to_numpy(continuous_actions)

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

    def _continuous_mle_immitation_loss(
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
        ), f"Inside _continuous_mle_immitation_loss: log std logits is none for type: {self.std_type}"

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
            -dist.log_prob(continuous_actions).sum(axis=-1).mean()
        )  # TODO: dist.entropy() to stop it from overfitting
        return loss

    def _continuous_naive_immitation_loss(
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
        ), f"Inside _continuous_naive_immitation_loss: log std logits is none for type: {self.std_type}"

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
        continuous_mean_logits, continuous_log_std_logits, discrete_logits = self.actor(
            x=observations, action_mask=action_mask, debug=False
        )
        continuous_immitation_loss = torch.zeros(1, device=self.device)
        discrete_immitation_loss = torch.zeros(1, device=self.device)

        if self.continuous_action_dim > 0 and continuous_actions is not None:
            if self.naive_immitation:
                continuous_immitation_loss = self._continuous_mle_immitation_loss(
                    continuous_mean_logits,
                    continuous_log_std_logits,
                    continuous_actions,
                )
            else:
                continuous_immitation_loss = self._continuous_naive_immitation_loss(
                    continuous_mean_logits,
                    continuous_log_std_logits,
                    continuous_actions,
                    0.1,
                )
        if self.discrete_action_dims is not None and discrete_actions is not None:
            discrete_immitation_loss = self._discrete_imitation_loss(
                discrete_logits, discrete_actions
            )

        loss = discrete_immitation_loss + continuous_immitation_loss
        self.optimizer.zero_grad()
        loss.backward()  # type:ignore  started as a float
        self.optimizer.step()

        if isinstance(discrete_immitation_loss, torch.Tensor):
            discrete_immitation_loss = discrete_immitation_loss.to("cpu").item()
        if isinstance(continuous_immitation_loss, torch.Tensor):
            continuous_immitation_loss = continuous_immitation_loss.to("cpu").item()
        return discrete_immitation_loss, continuous_immitation_loss

    def utility_function(self, observations, actions=None):
        if not torch.is_tensor(observations):
            observations = torch.tensor(observations, dtype=torch.float).to(self.device)
        if actions is not None:
            return self.critic(observations, actions)
        else:
            return self.critic(observations)
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action=None):
        return self.critic(obs)

    def _get_disc_log_probs_entropy(self, logits, actions):
        log_probs = torch.zeros_like(actions, dtype=torch.float)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
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

        dist = torch.distributions.Normal(
            loc=logits, scale=torch.clip(torch.exp(lstd), min=1e-6)
        )
        if self.action_clamp_type == "tanh":
            # dist = TransformedDistribution(dist, TanhTransform())
            # print("actions were tanhed so we need to get form raw to dist activations")
            # print(actions)
            activations = minmaxnorm(actions, self.min_actions, self.max_actions)
            eps = 1e-6
            activations = torch.clamp(activations, -1 + eps, 1 - eps)
            # print(f"normed actions: {activations}")
            activations = torch.atanh(activations)
            # print(f"inverse tanh actions: {activations}")
        else:
            activations = actions

        log_probs = dist.log_prob(activations)
        # correction = torch.zeros_like(log_probs, dtype=torch.float)

        if self.action_clamp_type == "tanh":
            log_probs -= 2 * (np.log(2) - activations - F.softplus(-2 * activations))
        if torch.min(log_probs) < -1000:
            print(
                f"{self.action_clamp_type} Warning: log_probs has very low values: {torch.min(log_probs)}. "
                "This might cause numerical instability."
            )
            input(f"print the rest? {self.action_clamp_type}")
            print(f"loc: {logits}, scale: {torch.exp(lstd)} actions: {activations}")
            log_probs = torch.clamp(log_probs, -1000, 2)
        return log_probs, dist.entropy().mean()

    def _get_probs_and_entropy(self, batch: FlexiBatch, agent_num):
        bm = None
        if batch.action_mask is not None:
            bm = batch.action_mask[agent_num]

        assert hasattr(
            batch, "obs"
        ), "Batch needs attribute 'obs' for PG stabalized get_probs_and_entropy to work"

        continuous_means, continuous_log_std_logits, discrete_logits = self.actor(
            x=batch.__getattr__(self.batch_name_map["obs"])[agent_num],
            action_mask=bm,  # type:ignore
        )
        old_disc_log_probs = 0
        old_disc_entropy = 0
        old_cont_log_probs = 0
        old_cont_entropy = 0

        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            assert (
                hasattr(batch, "discrete_actions")
                and batch.__getattr__(self.batch_name_map["discrete_actions"])
                is not None
            ), "Batch does not have attribute 'discrete_actions' but model has discrete_action_dims"
            old_disc_log_probs = []
            old_disc_entropy = 0
            for head in range(len(self.discrete_action_dims)):
                odlp, ode = self._get_disc_log_probs_entropy(
                    logits=discrete_logits[head],
                    actions=batch.__getattr__(self.batch_name_map["discrete_actions"])[
                        agent_num
                    ][
                        :, head
                    ],  # type:ignore
                )
                old_disc_log_probs.append(odlp)
                old_disc_entropy += ode

        if self.continuous_action_dim > 0:
            assert (
                hasattr(batch, "continuous_actions")
                and batch.__getattr__("continuous_actions") is not None
            ), "Batch does not have attribute 'continuous_actions' but model has discrete_action_dims"
            old_cont_log_probs, old_cont_entropy = self._get_cont_log_probs_entropy(
                logits=continuous_means,
                actions=batch.__getattr__(self.batch_name_map["continuous_actions"])[
                    agent_num
                ],  # type:ignore
                lstd_logits=continuous_log_std_logits,
            )
        return (
            old_disc_log_probs,
            old_disc_entropy,
            old_cont_log_probs,
            old_cont_entropy,
        )

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
        # print(f"    V_current: {V_current.shape}, G[indices] {G[indices].shape}")
        # input()
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
        else:
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

        cont_log_probs, cont_entropy = self._get_cont_log_probs_entropy(
            logits=action_means,
            actions=actions,
            lstd_logits=action_log_std,
        )
        if self.ppo_clip > 0:
            logratio = (
                cont_log_probs
                - old_log_probs  # batch.continuous_log_probs[agent_num, indices]
            )
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
        return actor_loss

    def _discrete_actor_loss(self, actions, log_probs, logits, advantages):
        actor_loss = torch.zeros(1, device=self.device)
        for head in range(actions.shape[-1]):
            dist = Categorical(logits=logits[head])  # TODO: th
            entropy = dist.entropy().mean()
            selected_log_probs = dist.log_prob(actions[:, head])
            if self.ppo_clip > 0:
                logratio = (
                    selected_log_probs
                    - log_probs[
                        :, head
                    ]  # batch.discrete_log_probs[agent_num, indices, head]
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
        return actor_loss

    def reinforcement_learn(
        self,
        batch: FlexiBatch,
        agent_num=0,
        critic_only=False,
        debug=False,
    ):
        if self.eval_mode:
            return 0, 0
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

        for epoch in range(self.n_epochs):
            if debug:
                print("  Starting epoch", epoch)
            bnum = 0

            while self.mini_batch_size * bnum < bsize:
                # Get Critic Loss
                bstart = self.mini_batch_size * bnum
                bend = min(bstart + self.mini_batch_size, bsize - 1)
                indices = mini_batch_indices[bstart:bend]
                bnum += 1
                if debug:
                    print(
                        f"    Mini batch: {bstart}:{bend}, Indices: {indices}, {len(indices)}"
                    )

                critic_loss = self._critic_loss(batch, indices, G, agent_num, debug)
                actor_loss = torch.zeros(1, device=self.device)
                # print(torch.abs(V_current - G[indices]).mean())
                if not critic_only:
                    mb_adv = advantages[torch.from_numpy(indices).to(self.device)]
                    continuous_means, continuous_log_std_logits, discrete_logits = (
                        self.actor(
                            x=batch.__getattr__(self.batch_name_map["obs"])[
                                agent_num, indices
                            ],
                        )
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
                        # dlp = []
                        # for head in range(len(self.discrete_action_dims)):
                        #     dlp.append(
                        #         batch.__getattr__(
                        #             self.batch_name_map["discrete_log_probs"]
                        #         )[head][agent_num, indices]
                        #     )
                        dlp = batch.__getattr__(
                            self.batch_name_map["discrete_log_probs"]
                        )[agent_num, indices]
                        actor_loss += self._discrete_actor_loss(
                            dact, dlp, discrete_logits, mb_adv
                        )

                    # print("actor")
                    # self.optimizer.zero_grad()
                    # loss = actor_loss
                    # loss.backward()
                    # self._print_grad_norm()
                    # print("critic")
                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss * self.critic_loss_coef
                loss.backward()
                # self._print_grad_norm()
                # print(self.actor_logstd)
                # print(self.actor_logstd.grad)
                # self._print_grad_norm()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        0.5,
                        error_if_nonfinite=True,
                        foreach=True,
                    )

                self.optimizer.step()

                avg_actor_loss += actor_loss.to("cpu").item()
                avg_critic_loss += critic_loss.to("cpu").item()
            avg_actor_loss /= nbatch
            avg_critic_loss /= nbatch
            # print(f"actor_loss: {actor_loss.item()}")

        avg_actor_loss /= self.n_epochs
        avg_critic_loss /= self.n_epochs
        # print(avg_actor_loss, critic_loss.item())
        return avg_actor_loss, avg_critic_loss

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
        for i in range(len(self.attrs)):
            self._dump_attr(
                self.__dict__[self.attrs[i]], checkpoint_path + f"/{self.attrs[i]}"
            )

    def load(self, checkpoint_path):
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"

        for i in range(len(self.attrs)):
            self.__dict__[self.attrs[i]] = self._load_attr(
                checkpoint_path + f"/{self.attrs[i]}"
            )
        self._get_torch_params(self.starting_actorlogstd)
        self.policy_loss = 5.0
        self.actor.load_state_dict(torch.load(checkpoint_path + "/PI"))
        self.critic.load_state_dict(torch.load(checkpoint_path + "/V"))
        self.actor_logstd = torch.load(checkpoint_path + "/actor_logstd")

    def __str__(self):
        st = ""
        for d in self.__dict__.keys():
            st += f"{d}: {self.__dict__[d]}"
        return st

    def reinforcement_learn_perf(
        self,
        batch: FlexiBatch,
        agent_num=0,
        critic_only=False,
        debug=False,
    ):
        __s = time.time()
        if self.eval_mode:
            return 0, 0
        if debug:
            print(f"Starting PG Reinforcement Learn for agent {agent_num}")
        # self.run_times = {"advantage":0.0,"aloss":0.0,"closs":0.0,"backward":0.0,"total_time"}
        _s = time.time()
        with torch.no_grad():
            G, advantages, values = self._calculate_advantages(batch, agent_num, debug)

        # print(G.shape)
        # input("hmm rl")
        assert isinstance(
            advantages, torch.Tensor
        ), "Advantages has to be a tensor but it isn't, maybe batch was not called with as_torch=True?"
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.run_times["advantage"] += time.time() - _s

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

        for epoch in range(self.n_epochs):
            if debug:
                print("  Starting epoch", epoch)
            bnum = 0

            while self.mini_batch_size * bnum < bsize:
                # Get Critic Loss
                bstart = self.mini_batch_size * bnum
                bend = min(bstart + self.mini_batch_size, bsize - 1)
                indices = mini_batch_indices[bstart:bend]
                bnum += 1
                if debug:
                    print(
                        f"    Mini batch: {bstart}:{bend}, Indices: {indices}, {len(indices)}"
                    )
                _s = time.time()
                critic_loss = self._critic_loss(batch, indices, G, agent_num, debug)
                self.run_times["critic_loss"] += time.time() - _s

                actor_loss = torch.zeros(1, device=self.device)
                # print(torch.abs(V_current - G[indices]).mean())
                if not critic_only:
                    mb_adv = advantages[torch.from_numpy(indices).to(self.device)]
                    _s = time.time()
                    continuous_means, continuous_log_std_logits, discrete_logits = (
                        self.actor(
                            x=batch.__getattr__(self.batch_name_map["obs"])[
                                agent_num, indices
                            ],
                        )
                    )
                    self.run_times["act"] += time.time() - _s

                    if self.continuous_action_dim > 0:
                        _s = time.time()
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
                        self.run_times["closs"] += time.time() - _s

                    if self.discrete_action_dims is not None:
                        _s = time.time()
                        dact = batch.__getattr__(
                            self.batch_name_map["discrete_actions"]
                        )[agent_num, indices]
                        dlp = batch.__getattr__(
                            self.batch_name_map["discrete_log_probs"]
                        )[agent_num, indices]
                        actor_loss += self._discrete_actor_loss(
                            dact, dlp, discrete_logits, mb_adv
                        )
                        self.run_times["dloss"] = time.time() - _s
                    # print("actor")
                    # self.optimizer.zero_grad()
                    # loss = actor_loss
                    # loss.backward()
                    # self._print_grad_norm()
                    # print("critic")
                _s = time.time()
                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss * self.critic_loss_coef
                loss.backward()
                # self._print_grad_norm()
                # print(self.actor_logstd)
                # print(self.actor_logstd.grad)
                # self._print_grad_norm()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        0.5,
                        error_if_nonfinite=True,
                        foreach=False,
                    )

                self.optimizer.step()

                self.run_times["backward"] += time.time() - _s

                avg_actor_loss += actor_loss.to("cpu").item()
                avg_critic_loss += critic_loss.to("cpu").item()
            avg_actor_loss /= nbatch
            avg_critic_loss /= nbatch
        avg_actor_loss /= self.n_epochs
        avg_critic_loss /= self.n_epochs

        self.run_times["tot"] += time.time() - __s
        return avg_actor_loss, avg_critic_loss
