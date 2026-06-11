from .Agent import ValueS, StochasticActor, Agent, QMixer, QS, VDNMixer
from .Util import T, minmaxnorm
import torch
from .buffers import RolloutBuffer, ReplayBuffer, RolloutBufferSamples, ReplayBufferSamples
from torch.distributions import Categorical
import numpy as np
import torch.nn as nn
import pickle
import os
import time
from torch.distributions import TransformedDistribution, TanhTransform
import torch.nn.functional as F
import collections

_LOG2 = float(np.log(2.0))


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
        importance_schedule=[0.0, 1.0, 1000],  # start_alpha, end_alpha, n_steps
        importance_from_grad=True,
        on_policy_mixer=True,
        logit_reg=0.05,
        relative_entropy_loss=0.05,
        wall_time=False,
        joint_kl_penalty=0.1,
        target_kl=0.1,
        use_kl_penalty=False,
        offline_critic_buffer=False,
    ):
        super(PG, self).__init__()
        config = locals()
        # Remove 'self' and other unwanted items
        config.pop("self")
        self.joint_kl_penalty = joint_kl_penalty
        self.target_kl = target_kl
        self.use_kl_penalty = use_kl_penalty
        self.config = config
        self.wall_time = wall_time
        self.load_from_checkpoint = load_from_checkpoint
        self.relative_entropy_loss = relative_entropy_loss
        if self.load_from_checkpoint is not None:
            self.load(self.load_from_checkpoint)
            return

        # Set up the params for Qmixed PPO
        self.importance_alpha = importance_schedule[0]
        self.importance_alpha_start = importance_schedule[0]
        self.importance_alpha_end = importance_schedule[1]
        self.importance_alpha_steps = importance_schedule[2]
        self.importance_step = 0
        self.importance_from_grad = importance_from_grad
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
        self.use_offline_critic_buffer = offline_critic_buffer

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
        if self.mix_type is not None and self.mix_type.lower() == "vdn":
            self.mix_type = "VDN"
            self.advantage_type = "qmix"
        elif self.mix_type is not None and self.mix_type.lower() == "qmix":
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

    def _init_offline_critic_buffer(self, n_envs: int = 1, n_agents: int = 1) -> None:
        # 5000 T-slots × n_envs × n_agents ≈ 5000 * n_envs total transitions.
        # full_gpu keeps the replay data on CUDA for zero-copy sampling.
        self._offline_buffer = ReplayBuffer(
            buffer_size=5000,
            obs_shape=(self.obs_dim,),
            action_dim=self.total_action_dims,
            device=self.device,
            n_envs=n_envs,
            n_agents=n_agents,
            full_gpu=True,
        )

    def _add_batch_to_offline_buffer(self, buffer: RolloutBuffer) -> None:
        T = buffer.pos
        if T < 2:
            return

        E, A = buffer.n_envs, buffer.n_agents
        # Lazy-init (or re-init if env count changed) so we never need n_envs at __init__ time.
        if not hasattr(self, '_offline_buffer') or self._offline_buffer.n_envs != E:
            self._init_offline_critic_buffer(n_envs=E, n_agents=A)

        obs = buffer.observations[:T]                          # [T, E, A, obs_dim]
        next_obs = torch.cat([obs[1:], obs[-1:]], dim=0)       # [T, E, A, obs_dim]

        # T-1 vectorised calls (one per timestep, all envs at once) instead of T×E×A scalar calls.
        for t in range(T - 1):
            self._offline_buffer.add(
                obs=obs[t],                    # [E, A, obs_dim]
                next_obs=next_obs[t],          # [E, A, obs_dim]
                action=buffer.actions[t],      # [E, A, act_dim]
                reward=buffer.rewards[t],      # [E, A]
                term=buffer.terminations[t],   # [E, A]
                trunc=buffer.truncations[t],   # [E, A]
            )

    def _offline_critic_loss(self) -> torch.Tensor:
        if not hasattr(self, '_offline_buffer') or self._offline_buffer.size() < self.mini_batch_size:
            return torch.zeros(1, device=self.device)

        samples = self._offline_buffer.sample(self.mini_batch_size)
        obs = samples.observations
        obs_ = samples.next_observations
        rewards = samples.rewards.squeeze(-1)
        terminated = samples.terminations.squeeze(-1)
        actions = samples.actions # [B, D]
        
        # Split actions back into discrete and continuous if needed for importance gathering
        d_actions = None
        c_actions = None
        if self.discrete_action_dims is not None:
            d_actions = actions[:, :len(self.discrete_action_dims)].long()
        if self.continuous_action_dim > 0:
            c_actions = actions[:, -self.continuous_action_dim:]

        values, d_adv, c_adv = self.critic(obs, policy_weights=self._critic_policy_weights(obs))
        if c_adv is not None:
            c_adv = torch.transpose(torch.stack(c_adv, dim=0), 0, 1)
        adv = self._gather_observed_advantages(d_adv, c_adv, d_actions, c_actions)
        Q = (self.mixer(adv, obs)[0] + values).squeeze(-1)

        with torch.no_grad():
            next_values, next_d_adv, next_c_adv = self.critic(obs_, policy_weights=self._critic_policy_weights(obs_))
            if self.on_policy_mixer:
                next_Q = next_values.squeeze(-1)
            else:
                if next_c_adv is not None:
                    next_c_adv = torch.transpose(torch.stack(next_c_adv, dim=0), 0, 1)
                next_adv = self._max_advantages(next_d_adv, next_c_adv)
                next_Q = (self.mixer(next_adv, obs_)[0] + next_values).squeeze(-1)

            target_Q = rewards + self.gamma * (1.0 - terminated) * next_Q

        return ((Q - target_Q) ** 2).mean()

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
            # QPLEX-style separation: the dueling V head is the *only* state-value
            # term, so the mixer must satisfy Mixer(0; s) = 0.  Disabling both the
            # internal bias b1 and the final bias b2 guarantees this (b2 alone is a
            # second V(s); b1 leaks through leaky_relu as leaky_relu(b1)*w2 != 0).
            # Mirrors the QS internal mixer (use_b1/b2 = not dueling).
            self.mixer = QMixer(
                self.total_action_dims, self.obs_dim, mixing_embed_dim=self.mixer_dim,
                use_b1=False, use_b2=False,
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
        if self.std_type == "stateless":
            init_val = torch.full(
                (self.continuous_action_dim,), 0.5493, device=self.device
            )
            self.actor_logstd = nn.Parameter(init_val, requires_grad=True)

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
        else:
            return np.array(x)

    def train_actions(self, observations, action_mask=None, step=False, debug=False):
        t = 0
        if self.wall_time:
            t = time.time()
        
        was_tensor = torch.is_tensor(observations)
        if not was_tensor:
            observations = T(observations, device=self.device, dtype=torch.float)
        
        # observations shape: [n_envs, n_agents, obs_dim] per [T,E,A,O] layout, or [B, obs_dim]
        # StochasticActor expects [batch, obs_dim]
        is_3d = (len(observations.shape) == 3)
        if is_3d:
            n_envs, n_agents, obs_dim = observations.shape
            obs_flat = observations.reshape(-1, obs_dim)
        else:
            obs_flat = observations

        if action_mask is not None and not torch.is_tensor(action_mask):
            action_mask = torch.tensor(
                action_mask, dtype=torch.float, device=self.device
            )

        if step:
            self.steps += 1
        if self.anneal_lr > 0:
            frac = max(1.0 - (self.steps - 1.0) / self.anneal_lr, 0.0001)
            lrnow = frac * self.lr
            self.optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            continuous_logits, continuous_log_std_logits, discrete_action_logits = (
                self.actor(x=obs_flat, action_mask=action_mask, debug=debug)
            )

            if continuous_log_std_logits is None and self.continuous_action_dim > 0:
                assert (
                    self.std_type == "stateless"
                ), "Log std logits should only be none if we don't want the actor producing them aka stateless"
                continuous_log_std_logits = self._smooth_clamp_logstd()
            
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

            # Unflatten results if input was 3D — output is [E, A, ...]
            if is_3d:
                if discrete_actions is not None:
                    discrete_actions = discrete_actions.reshape(n_envs, n_agents, -1)
                if continuous_actions is not None:
                    continuous_actions = continuous_actions.reshape(n_envs, n_agents, -1)
                if discrete_log_probs is not None:
                    discrete_log_probs = discrete_log_probs.reshape(n_envs, n_agents, -1)
                if continuous_log_probs is not None:
                    continuous_log_probs = continuous_log_probs.reshape(n_envs, n_agents, -1)

        if self.wall_time:
            t = time.time() - t

        values_flat = self.expected_V(obs_flat).detach()
        if is_3d:
            values = values_flat.reshape(n_envs, n_agents)
        else:
            values = values_flat
            
        def _maybe_numpy(x):
            if was_tensor:
                return x
            return self._to_numpy(x)

        act = {
            "discrete_actions": _maybe_numpy(discrete_actions),
            "continuous_actions": _maybe_numpy(continuous_actions),
            "discrete_log_probs": _maybe_numpy(discrete_log_probs),
            "continuous_log_probs": _maybe_numpy(continuous_log_probs),
            "values": _maybe_numpy(values),
            "act_time": t,
        }
        return act

    def stable_greedy(self, obs, legal_action):
        ad = self.train_actions(
            observations=obs, action_mask=legal_action, step=False, debug=False
        )
        adiscrete, acontinuous = None, None
        if ad["discrete_actions"] is not None:
            adiscrete = torch.tensor(ad["discrete_actions"], device=self.device)
        if ad["continuous_actions"] is not None:
            acontinuous = torch.tensor(ad["continuous_actions"], device=self.device)
        return adiscrete, acontinuous

    def ego_actions(self, observations, action_mask=None):
        with torch.no_grad():
            continuous_logits, continuous_log_std_logits, discrete_action_logits = (
                self.actor(x=observations, action_mask=action_mask, debug=False)
            )
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

    def utility_function(self, observations, actions=None):
        if not torch.is_tensor(observations):
            observations = torch.tensor(
                observations, dtype=torch.float, device=self.device
            )
        if actions is not None:
            return self.critic(observations, actions)
        else:
            return self.critic(observations)

    def expected_V(self, obs, legal_action=None):
        is_3d = (len(obs.shape) == 3)
        if is_3d:
            n_envs, n_agents, obs_dim = obs.shape
            obs_flat = obs.reshape(-1, obs_dim)
        else:
            obs_flat = obs

        if self.mix_type is None:
            v_flat = self.critic(obs_flat).squeeze(-1)
        else:
            values, disc_advantages, cont_advantages = self.critic(obs_flat)
            v_flat = values.squeeze(-1)

        if is_3d:
            return v_flat.reshape(n_envs, n_agents)
        return v_flat

    def _smooth_clamp_logstd(self):
        low, high = self.log_std_clamp_range
        return low + 0.5 * (high - low) * (torch.tanh(self.actor_logstd) + 1.0)

    def _get_cont_log_probs_entropy(
        self, logits, actions, lstd_logits: torch.Tensor | None = None
    ):
        lstd = -1.0
        if self.actor_logstd is not None:
            lstd = self._smooth_clamp_logstd().expand_as(logits)
        else:
            assert (
                lstd_logits is not None
            ), "If the actor doesnt generate logits then it needs to have a global logstd"
            lstd = lstd_logits.expand_as(logits)

        if self.action_clamp_type == "tanh":
            dist = torch.distributions.Normal(loc=logits, scale=torch.exp(lstd))
            activations = minmaxnorm(actions, self.min_actions, self.max_actions)
            activations = torch.clamp(activations, -(1.0 - 1e-6), 1.0 - 1e-6)
            activations = torch.atanh(activations)
        else:
            dist = torch.distributions.Normal(loc=logits, scale=torch.exp(lstd))
            activations = actions

        log_probs = dist.log_prob(activations).sum(dim=-1)

        if self.action_clamp_type == "tanh":
            log_probs -= 2 * (_LOG2 - activations - F.softplus(-2 * activations)).sum(
                dim=-1
            )

        if torch.min(log_probs) < -100:
            self.end_early = True
            # print(f"{self.action_clamp_type} Warning: log_probs very low: {torch.min(log_probs)}")
            
        eloss = dist.entropy().mean()
        return log_probs, eloss

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
            logratio = cont_log_probs - old_log_probs
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
        if not torch.isnan(al):
            actor_loss += al
        self.result_dict["c_entropy"] += cont_entropy.item()
        return actor_loss

    def _discrete_actor_loss(self, actions, log_probs, logits, advantages):
        actor_loss = 0.0
        for head in range(actions.shape[-1]):
            dist = Categorical(logits=logits[head])
            entropy = dist.entropy().mean()
            selected_log_probs = dist.log_prob(actions[:, head])
            if self.ppo_clip > 0:
                old_lp = log_probs[:, head]
                logratio = selected_log_probs - old_lp
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
        buffer: RolloutBuffer,
        last_values: torch.Tensor,
        last_terminations: np.ndarray,
        last_truncations: np.ndarray,
        critic_only=False,
        debug=False,
    ):
        t = 0
        if self.wall_time:
            t = time.time()
        if self.eval_mode:
            return {"rl_actor_loss": 0, "rl_critic_loss": 0, "rl_time": 0}

        self.result_dict = {
            "rl_actor_loss": 0,
            "rl_critic_loss": 0,
            "d_entropy": 0,
            "c_entropy": 0,
            "c_std": 0,
            "rl_time": t,
        }

        if self.mix_type == "QMIX" or self.mix_type == "VDN":
            return self._mix_reinforcement_learn(buffer, last_values, last_terminations, last_truncations, critic_only, debug)

        # 1. Compute Advantages
        with torch.no_grad():
            buffer.compute_returns_and_advantage(
                last_values, last_terminations, last_truncations,
                get_value_fn=lambda obs: self.expected_V(obs)
            )
        
        # 2. Mini-batch Training
        avg_actor_loss = 0
        avg_critic_loss = 0
        bnum = 0
        
        for epoch in range(self.n_epochs):
            for samples in buffer.get(batch_size=self.mini_batch_size):
                bnum += 1
                obs = samples.observations
                actions = samples.actions
                old_values = samples.old_values.reshape(-1)
                old_log_prob = samples.old_log_prob # [B, log_probs_dim]
                advantages = samples.advantages.reshape(-1)
                returns = samples.returns.reshape(-1)

                if self.norm_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Critic Loss with optional value clipping
                values = self.critic(obs).squeeze(-1)
                if self.value_clip > 0:
                    v_clip = old_values + (values - old_values).clamp(-self.value_clip, self.value_clip)
                    critic_loss = 0.5 * torch.max(F.mse_loss(values, returns), F.mse_loss(v_clip, returns))
                else:
                    critic_loss = 0.5 * F.mse_loss(values, returns)

                # Actor Loss
                actor_loss = 0.0
                continuous_means, continuous_log_std_logits, discrete_logits = self.actor(obs)
                
                # Split old_log_prob back into discrete and continuous parts
                lp_idx = 0
                if self.continuous_action_dim > 0:
                    c_old_lp = old_log_prob[:, lp_idx]
                    lp_idx += 1
                    c_actions = actions[:, -self.continuous_action_dim:]
                    actor_loss += self._continuous_actor_loss(
                        continuous_means, continuous_log_std_logits, c_old_lp, advantages, c_actions
                    )
                
                if self.discrete_action_dims is not None:
                    d_old_lp = old_log_prob[:, lp_idx:]
                    d_actions = actions[:, :len(self.discrete_action_dims)].long()
                    actor_loss += self._discrete_actor_loss(
                        d_actions, d_old_lp, discrete_logits, advantages
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss = actor_loss + critic_loss * self.critic_loss_coef
                loss.backward()

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()

                avg_actor_loss += actor_loss.item()
                avg_critic_loss += critic_loss.item()

        avg_actor_loss /= bnum
        avg_critic_loss /= bnum
        if self.wall_time:
            self.result_dict["rl_time"] = time.time() - t
        self.result_dict["rl_actor_loss"] = avg_actor_loss
        self.result_dict["rl_critic_loss"] = avg_critic_loss
        return self.result_dict

    def _get_dists(self, obs):
        """Return current-policy distributions for KL computation (no grad)."""
        d_dists, c_dist = None, None
        continuous_means, continuous_log_std_logits, discrete_logits = self.actor(obs)
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            d_dists = []
            for logits in discrete_logits:
                d_dists.append(torch.distributions.Categorical(logits=logits))
        if self.continuous_action_dim > 0:
            lstd = (self._smooth_clamp_logstd().expand_as(continuous_means)
                    if self.actor_logstd is not None
                    else continuous_log_std_logits.expand_as(continuous_means))
            c_dist = torch.distributions.Normal(loc=continuous_means, scale=torch.exp(lstd))
        return d_dists, c_dist

    def _mix_actor_loss(self, old_log_probs, new_log_probs, advantages, entropy,
                        old_d_dists, new_d_dists, old_c_dists, new_c_dists):
        """PPO clip loss with per-head advantages and joint KL penalty (legacy-equivalent)."""
        logratio = new_log_probs - old_log_probs
        ratio = torch.exp(logratio)
        clip_ratio = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        policy_loss = -(torch.min(advantages * ratio, advantages * clip_ratio).sum(-1)).mean()

        ent = 0.0
        if isinstance(entropy, torch.Tensor): ent = entropy.mean()
        policy_loss = policy_loss - self.entropy_loss * ent

        # Actual KL divergence using distribution objects
        kl_div = None
        if old_d_dists is not None and new_d_dists is not None:
            for old_d, new_d in zip(old_d_dists, new_d_dists):
                kl_piece = torch.distributions.kl_divergence(old_d, new_d)
                kl_div = kl_piece if kl_div is None else kl_div + kl_piece
        if old_c_dists is not None and new_c_dists is not None:
            kl_piece = torch.distributions.kl_divergence(old_c_dists, new_c_dists).sum(-1)
            kl_div = kl_piece if kl_div is None else kl_div + kl_piece

        joint_kl = torch.tensor(0.0, device=self.device)
        if kl_div is not None:
            joint_kl = kl_div.mean()
            policy_loss = policy_loss + self.joint_kl_penalty * joint_kl
            if joint_kl.item() > self.target_kl:
                self.joint_kl_penalty = min(self.joint_kl_penalty * 1.5, 100.0)
            elif joint_kl.item() < self.target_kl / 1.5:
                self.joint_kl_penalty = max(self.joint_kl_penalty / 1.5, 1e-4)

        return policy_loss, joint_kl

    def _mix_critic_only(self, buffer: RolloutBuffer, last_values: torch.Tensor,
                         last_terminations: np.ndarray, last_truncations: np.ndarray):
        if self.use_offline_critic_buffer:
            # Offline TD critic update — mixer stays in the loss so it is trained too.
            self._add_batch_to_offline_buffer(buffer)
            critic_loss = self._offline_critic_loss()
            self.optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                if self.mixer is not None and self.mix_type == "QMIX":
                    torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 0.5)
            self.optimizer.step()
            self.result_dict["rl_critic_loss"] = critic_loss.item()
            return self.result_dict

        obs = buffer.observations[:buffer.pos]
        actions = buffer.actions[:buffer.pos]
        rewards = buffer.rewards[:buffer.pos]
        terminated = buffer.terminations[:buffer.pos]
        truncated = buffer.truncations[:buffer.pos]
        T_steps, E, A, _ = obs.shape

        obs_f = obs.reshape(-1, self.obs_dim).to(self.device)
        d_actions_f = actions[:, :, :, :len(self.discrete_action_dims)].reshape(-1, len(self.discrete_action_dims)).long().to(self.device) if self.discrete_action_dims else None
        c_actions_f = actions[:, :, :, -self.continuous_action_dim:].reshape(-1, self.continuous_action_dim).to(self.device) if self.continuous_action_dim > 0 else None

        values_f, d_adv_f, c_adv_f = self.critic(obs_f, policy_weights=self._critic_policy_weights(obs_f))
        if c_adv_f is not None:
            c_adv_f = torch.transpose(torch.stack(c_adv_f, dim=0), 0, 1)
        adv_f = self._gather_observed_advantages(d_adv_f, c_adv_f, d_actions_f, c_actions_f)
        Q_f = (self.mixer(adv_f, obs_f)[0] + values_f).squeeze(-1)

        V = values_f.detach().reshape(T_steps, E, A)
        bootstrap_values = last_values.reshape(E, A)
        term_t = torch.as_tensor(terminated, device=self.device)
        trunc_t = torch.as_tensor(truncated, device=self.device)

        with torch.no_grad():
            G_critic, _ = self._weighted_gae(
                rewards=rewards, values=V, bootstrap_values=bootstrap_values,
                terminated=term_t, truncated=trunc_t,
                advantage_weights=torch.ones_like(V).unsqueeze(-1),
                gamma=self.gamma, gae_lambda=self.gae_lambda,
                final_observations=buffer.final_observations,
                get_value_fn=lambda o: self.expected_V(o),
            )
        G_critic_flat = G_critic.squeeze(-1).reshape(-1).to(self.device)
        critic_loss = F.mse_loss(Q_f, G_critic_flat)
        self.optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            if self.mixer is not None and self.mix_type == "QMIX":
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 0.5)
        self.optimizer.step()
        self.result_dict["rl_critic_loss"] = critic_loss.item()
        return self.result_dict

    def _weighted_gae(
        self,
        rewards: torch.Tensor,           # [T, E, A]
        values: torch.Tensor,            # [T, E, A]
        bootstrap_values: torch.Tensor,  # [E, A]
        terminated: torch.Tensor,        # [T, E, A]
        truncated: torch.Tensor,         # [T, E, A]
        advantage_weights: torch.Tensor, # [T, E, A, D]
        gamma=0.99,
        gae_lambda=0.95,
        final_observations=None,
        get_value_fn=None,
    ):
        rewards = rewards.to(self.device)
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)

        T_steps, E, A, D = advantage_weights.shape
        advantages = torch.zeros((T_steps, E, A, D), device=self.device)
        last_gae_lam = torch.zeros((E, A, D), device=self.device)

        for step in reversed(range(T_steps)):
            if step == T_steps - 1:
                next_non_terminal = 1.0 - terminated[step]
                next_episode_continuation = 1.0 - torch.clamp(terminated[step] + truncated[step], 0.0, 1.0)
                next_values = bootstrap_values.clone()
            else:
                next_non_terminal = 1.0 - terminated[step]
                next_episode_continuation = 1.0 - torch.clamp(terminated[step] + truncated[step], 0.0, 1.0)
                next_values = values[step + 1].clone()

            # Handle truncation bootstrapping — keys are (t_step, e_idx, a_idx)
            if get_value_fn is not None and final_observations is not None:
                for (t_step, env_idx, agent_idx), final_obs in final_observations.items():
                    if t_step == step:
                        next_values[env_idx, agent_idx] = get_value_fn(final_obs.to(self.device))

            # delta [E, A]
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            # weighted_delta [E, A, D]
            weighted_delta = delta.unsqueeze(-1) * advantage_weights[step]

            last_gae_lam = weighted_delta + gamma * gae_lambda * next_episode_continuation.unsqueeze(-1) * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values.unsqueeze(-1)
        return returns, advantages

    def _mix_reinforcement_learn(
        self,
        buffer: RolloutBuffer,
        last_values: torch.Tensor,
        last_terminations: np.ndarray,
        last_truncations: np.ndarray,
        critic_only=False,
        debug=False,
    ):
        assert self.mixer is not None, "Mixer required for QMIX/VDN PPO"
        if critic_only:
            return self._mix_critic_only(buffer, last_values, last_terminations, last_truncations)
        t = 0
        if self.wall_time:
            t = time.time()

        if self.use_offline_critic_buffer:
            self._add_batch_to_offline_buffer(buffer)

        # Pull all data from buffer [T, E, A, ...]
        obs = buffer.observations[:buffer.pos]
        actions = buffer.actions[:buffer.pos]
        rewards = buffer.rewards[:buffer.pos]
        terminated = buffer.terminations[:buffer.pos]
        truncated = buffer.truncations[:buffer.pos]

        T_steps, E, A, _ = obs.shape

        # Pull discrete and continuous actions
        d_actions = None
        c_actions = None
        if self.discrete_action_dims is not None:
            d_actions = actions[:, :, :, :len(self.discrete_action_dims)].long()
        if self.continuous_action_dim > 0:
            c_actions = actions[:, :, :, -self.continuous_action_dim:]

        # Calculate Mixer Gradients for Importance
        with torch.no_grad():
            # Flatten to [T*E*A, ...] for critic call. Centre advantages by the
            # current policy (E_pi[A_h]=0) so V targets V^pi (AppendixA Asm. 2).
            obs_f = obs.reshape(-1, self.obs_dim).to(self.device)
            values_f, d_adv_f, c_adv_f = self.critic(obs_f, policy_weights=self._critic_policy_weights(obs_f))

            if c_adv_f is not None:
                c_adv_f = torch.transpose(torch.stack(c_adv_f, dim=0), 0, 1)

            # Reshape actions for gather
            d_actions_f = d_actions.reshape(-1, len(self.discrete_action_dims)).to(self.device) if d_actions is not None else None
            c_actions_f = c_actions.reshape(-1, self.continuous_action_dim).to(self.device) if c_actions is not None else None

            adv_f = self._gather_observed_advantages(d_adv_f, c_adv_f, d_actions_f, c_actions_f)

        grad_free_adv = adv_f.detach()
        grad_free_adv.requires_grad = True
        __q, adv_grad = self.mixer(grad_free_adv, obs_f, with_grad=True)
        self.mixer.zero_grad()

        with torch.no_grad():
            V = values_f.reshape(T_steps, E, A)

            # Weighted GAE
            if self.importance_from_grad:
                scaled_importance = (grad_free_adv * adv_grad).reshape(T_steps, E, A, -1)
            else:
                raw_importance = self._gather_importance(d_adv_f, c_adv_f)
                scaled_importance = (raw_importance * adv_grad).reshape(T_steps, E, A, -1)

            self._update_importance()
            abs_imp = scaled_importance.abs() + 1e-8
            powered = abs_imp**self.importance_alpha
            scaled_importance = powered / powered.sum(dim=-1, keepdim=True)

            bootstrap_values = last_values.reshape(E, A)
            term_t = torch.as_tensor(terminated, device=self.device)
            trunc_t = torch.as_tensor(truncated, device=self.device)
            
            G, gae = self._weighted_gae(
                rewards=rewards,
                values=V,
                bootstrap_values=bootstrap_values,
                terminated=term_t,
                truncated=trunc_t,
                advantage_weights=scaled_importance,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                final_observations=buffer.final_observations,
                get_value_fn=lambda o: self.expected_V(o)
            )
            
            # Global returns for critic — same D as scaled_importance; each head gets its own G
            G_critic, _ = self._weighted_gae(
                rewards=rewards,
                values=V,
                bootstrap_values=bootstrap_values,
                terminated=term_t,
                truncated=trunc_t,
                advantage_weights=torch.ones_like(scaled_importance),
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                final_observations=buffer.final_observations,
                get_value_fn=lambda o: self.expected_V(o)
            )
            # G_critic shape: [T, E, A, D]

        # Move all flat training arrays to device once — avoids per-mini-batch H2D transfers
        obs_flat      = obs.reshape(-1, self.obs_dim).to(self.device)
        actions_flat  = actions.reshape(-1, self.total_action_dims).to(self.device)
        gae_flat      = gae.reshape(-1, self.total_action_dims).to(self.device)
        G_critic_flat = G_critic.reshape(-1, self.total_action_dims).to(self.device)

        # Get old log probs for PPO clip (d/c_actions_f already on device from above)
        with torch.no_grad():
            old_lp_flat, _, _, _, _, _ = self._log_probs_per_dim(obs_flat, d_actions_f, c_actions_f)

        # Compute importance metrics for analysis
        importance_per_dim = scaled_importance.mean(dim=0).reshape(-1).detach().cpu().numpy()
        importance_raw = scaled_importance.detach().cpu().numpy()

        indices = np.arange(obs_flat.shape[0])
        avg_actor_loss = 0.0
        avg_critic_loss = 0.0
        avg_joint_kl = 0.0
        bnum = 0

        if self.norm_advantages:
            gae_flat = (gae_flat - gae_flat.mean()) / (gae_flat.std() + 1e-8)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            # Snapshot old-policy parameters once per epoch (one full actor forward),
            # then create per-mini-batch distributions cheaply by slicing the tensors.
            with torch.no_grad():
                old_c_means_full, old_c_lstd_full, old_d_logits_full = self.actor(obs_flat)
                if old_c_lstd_full is None and self.continuous_action_dim > 0:
                    old_c_lstd_full = self._smooth_clamp_logstd().expand_as(old_c_means_full)

            dist_step = 0
            for start in range(0, len(indices), self.mini_batch_size):
                bnum += 1
                mb_idx = indices[start:start+self.mini_batch_size]

                # All arrays already on device — plain indexing, no H2D transfer
                mb_obs      = obs_flat[mb_idx]
                mb_actions  = actions_flat[mb_idx]
                mb_gae      = gae_flat[mb_idx]
                mb_G_critic = G_critic_flat[mb_idx]
                mb_old_lp   = old_lp_flat[mb_idx]

                # Build old distributions for this mini-batch by slicing cached logits
                old_d_dists_mb = (
                    [torch.distributions.Categorical(logits=lg[mb_idx]) for lg in old_d_logits_full]
                    if old_d_logits_full is not None else None
                )
                if self.continuous_action_dim > 0:
                    old_c_lstd_mb = old_c_lstd_full[mb_idx].expand_as(old_c_means_full[mb_idx])
                    old_c_dist_mb = torch.distributions.Normal(
                        loc=old_c_means_full[mb_idx], scale=torch.exp(old_c_lstd_mb)
                    )
                else:
                    old_c_dist_mb = None

                mb_d_act = mb_actions[:, :len(self.discrete_action_dims)].long() if self.discrete_action_dims else None
                mb_c_act = mb_actions[:, -self.continuous_action_dim:] if self.continuous_action_dim > 0 else None

                # Critic Loss — both branches route through the mixer so the
                # critic AND mixer parameters receive gradient signal.
                if self.use_offline_critic_buffer:
                    # 1-step TD from the internal replay buffer; QMIX stays in the
                    # loop so its hypernetwork parameters are trained.
                    critic_loss = self._offline_critic_loss()
                else:
                    # On-policy: Q_tot = mixer(observed per-head advantages; s) + V(s)
                    # regresses against the global GAE return (G_critic uses ones
                    # weights, so every head holds the same scalar return).
                    mb_values, mb_d_adv, mb_c_adv = self.critic(mb_obs, policy_weights=self._critic_policy_weights(mb_obs))
                    if mb_c_adv is not None:
                        mb_c_adv = torch.transpose(torch.stack(mb_c_adv, dim=0), 0, 1)
                    mb_adv_gathered = self._gather_observed_advantages(mb_d_adv, mb_c_adv, mb_d_act, mb_c_act)
                    mb_Q_tot = (self.mixer(mb_adv_gathered, mb_obs)[0] + mb_values).squeeze(-1)  # [B]
                    critic_loss = F.mse_loss(mb_Q_tot, mb_G_critic[:, 0])  # [B] vs [B]

                # Actor Loss — uses actual KL via _mix_actor_loss
                (
                    mb_new_lp,
                    mb_d_entropy,
                    mb_c_entropy,
                    mb_logit_reg,
                    mb_d_dist,
                    mb_c_dist,
                ) = self._log_probs_per_dim(mb_obs, mb_d_act, mb_c_act)

                mb_entropy = 0.0
                if isinstance(mb_d_entropy, torch.Tensor): mb_entropy = mb_entropy + mb_d_entropy
                if isinstance(mb_c_entropy, torch.Tensor): mb_entropy = mb_entropy + mb_c_entropy

                actor_loss, joint_kl = self._mix_actor_loss(
                    mb_old_lp, mb_new_lp, mb_gae, mb_entropy,
                    old_d_dists_mb, mb_d_dist,
                    old_c_dist_mb, mb_c_dist,
                )
                actor_loss = actor_loss + mb_logit_reg

                self.optimizer.zero_grad(set_to_none=True)
                loss = self.value_loss_coef * critic_loss + actor_loss
                loss.backward()
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()

                avg_actor_loss += actor_loss.item()
                avg_critic_loss += critic_loss.item()
                avg_joint_kl += joint_kl.item() if isinstance(joint_kl, torch.Tensor) else float(joint_kl)
                dist_step += 1

        if self.wall_time:
            self.result_dict["rl_time"] = time.time() - t
        self.result_dict["rl_actor_loss"] = avg_actor_loss / bnum
        self.result_dict["rl_critic_loss"] = avg_critic_loss / bnum
        self.result_dict["joint_kl"] = avg_joint_kl / bnum
        self.result_dict["importance_per_dim"] = importance_per_dim
        self.result_dict["importance_raw"] = importance_raw
        return self.result_dict

    def _bin_continuous_actions(self, c_actions):
        assert (self.min_actions is not None and self.max_actions is not None), "Can't bin actions without bounds"
        n_bins = 5
        min_actions = self.min_actions.unsqueeze(0)
        max_actions = self.max_actions.unsqueeze(0)
        bin_width = (max_actions - min_actions) / (n_bins - 1)
        bin_indices = torch.round((c_actions - min_actions) / bin_width)
        bin_indices = bin_indices.clamp(0, n_bins - 1)
        return bin_indices.long()

    def _critic_policy_weights(self, obs):
        """Current-policy probabilities over each head's bins, used to centre the
        dueling critic's advantages by E_pi[A_h] = 0 instead of the uniform mean.

        This enforces AppendixA Assumption 2 (on-policy advantage normalisation),
        which is what lets the dueling V head target the true on-policy value
        V^pi(s) (QPLEX-style: value held separate, mixer acts on advantages).
        The weights are stop-gradient (constants w.r.t. the policy parameters).
        """
        with torch.no_grad():
            c_means, c_lstd_logits, d_logits = self.actor(obs)
            disc = None
            if self.discrete_action_dims is not None:
                disc = [torch.softmax(lg, dim=-1) for lg in d_logits]
            cont = None
            if self.continuous_action_dim > 0:
                cont = self._continuous_bin_probs(c_means, c_lstd_logits)
        return {"discrete": disc, "continuous": cont}

    def _continuous_bin_probs(self, c_means, c_lstd_logits):
        """Probability mass the (squashed) Gaussian policy assigns to each
        advantage bin, per continuous dim — mirrors QS's binning so the centering
        weight matches the head it centres."""
        if self.actor_logstd is not None:
            lstd = self._smooth_clamp_logstd().expand_as(c_means)
        else:
            lstd = c_lstd_logits.expand_as(c_means)
        std = torch.exp(lstd)
        bins = self.critic.c_action_bins  # one entry per continuous dim
        B = c_means.shape[0]
        probs = []
        for i in range(self.continuous_action_dim):
            n_bins = bins[i]
            mn = self.min_actions[i]
            mx = self.max_actions[i]
            bw = (mx - mn) / (n_bins - 1)
            centers = mn + torch.arange(n_bins, device=self.device, dtype=c_means.dtype) * bw
            edges = (centers[:-1] + centers[1:]) * 0.5  # interior edges, action space
            if self.action_clamp_type == "tanh":
                z = torch.atanh(torch.clamp(minmaxnorm(edges, mn, mx), -(1 - 1e-6), 1 - 1e-6))
            else:
                z = edges
            dist = torch.distributions.Normal(loc=c_means[:, i:i + 1], scale=std[:, i:i + 1])
            cdf = dist.cdf(z.unsqueeze(0))  # [B, n_bins-1]
            lo = torch.zeros(B, 1, device=self.device, dtype=cdf.dtype)
            hi = torch.ones(B, 1, device=self.device, dtype=cdf.dtype)
            cdf_full = torch.cat([lo, cdf, hi], dim=-1)  # [B, n_bins+1]
            probs.append(cdf_full[:, 1:] - cdf_full[:, :-1])  # [B, n_bins]
        return probs

    def _gather_observed_advantages(self, d_adv, c_adv, d_actions, c_actions):
        n_d = len(d_adv) if d_adv is not None else 0
        n_c = c_adv.shape[1] if c_adv is not None else 0
        ref = d_adv[0] if d_adv is not None else c_adv
        out = ref.new_zeros(ref.shape[0], n_c + n_d)
        # Continuous first — must match [cont, disc] ordering of _log_probs_per_dim
        if c_adv is not None:
            c_indices = self._bin_continuous_actions(c_actions)
            out[:, :n_c] = c_adv.gather(dim=-1, index=c_indices.unsqueeze(-1)).squeeze(-1)
        for h in range(n_d):
            out[:, n_c + h] = d_adv[h].gather(dim=-1, index=d_actions[:, h].unsqueeze(-1)).squeeze(-1)
        return out

    def _max_advantages(self, d_adv, c_adv):
        n_d = len(d_adv) if d_adv is not None else 0
        n_c = c_adv.shape[1] if c_adv is not None else 0
        ref = d_adv[0] if d_adv is not None else c_adv
        out = ref.new_zeros(ref.shape[0], n_c + n_d)
        if c_adv is not None:
            out[:, :n_c] = c_adv.max(dim=-1).values
        for h in range(n_d):
            out[:, n_c + h] = d_adv[h].max(dim=-1).values
        return out

    def _gather_importance(self, d_adv, c_adv):
        n_d = len(d_adv) if d_adv is not None else 0
        n_c = c_adv.shape[1] if c_adv is not None else 0
        ref = d_adv[0] if d_adv is not None else c_adv
        out = ref.new_zeros(ref.shape[0], n_c + n_d)
        if c_adv is not None:
            out[:, :n_c] = c_adv.max(dim=-1).values - c_adv.min(dim=-1).values
        for h in range(n_d):
            adv_h = d_adv[h].detach()
            out[:, n_c + h] = adv_h.max(dim=-1).values - adv_h.min(dim=-1).values
        return out

    def _update_importance(self):
        frac = min(self.importance_step / self.importance_alpha_steps, 1.0)
        self.importance_alpha = self.importance_alpha_start * (1.0 - frac) + self.importance_alpha_end * frac
        self.importance_step += 1

    def _log_probs_per_dim(self, obs, d_actions, c_actions):
        continuous_means, continuous_log_std_logits, discrete_logits = self.actor(obs)
        lp = []
        c_entropy = 0
        c_dist = None
        if self.continuous_action_dim > 0:
            lstd = self._smooth_clamp_logstd().expand_as(continuous_means) if self.actor_logstd is not None else continuous_log_std_logits.expand_as(continuous_means)
            dist = torch.distributions.Normal(loc=continuous_means, scale=torch.exp(lstd))
            c_dist = dist
            if self.action_clamp_type == "tanh":
                activations = minmaxnorm(c_actions, self.min_actions, self.max_actions)
                activations = torch.clamp(activations, -(1.0 - 1e-6), 1.0 - 1e-6)
                activations = torch.atanh(activations)
            else:
                activations = c_actions
            log_probs = dist.log_prob(activations)
            if self.action_clamp_type == "tanh":
                log_probs -= 2 * (_LOG2 - activations - F.softplus(-2 * activations))
            lp.append(log_probs)
            c_entropy = dist.entropy().sum(-1)

        d_entropy = 0
        d_dists = None
        if self.discrete_action_dims:
            d_dists = []
            discrete_lp_cols = []
            for i, logits in enumerate(discrete_logits):
                dist = Categorical(logits=logits)
                d_dists.append(dist)
                discrete_lp_cols.append(dist.log_prob(d_actions[:, i]))
                d_entropy += dist.entropy()
            lp.append(torch.stack(discrete_lp_cols, dim=-1))
            
        lp = torch.cat(lp, dim=-1)
        logit_reg = 0
        if self.continuous_action_dim > 0:
            logit_reg = (continuous_means[torch.abs(continuous_means) > 4.0] ** 2).mean()
        
        return lp, d_entropy, c_entropy, logit_reg, d_dists, c_dist

    def imitation_learn(self, observations, continuous_actions, discrete_actions, action_mask=None, debug=False):
        # PG doesn't explicitly support imitation learning in this context, returning zero loss
        return {"rl_actor_loss": 0, "rl_critic_loss": 0}

    def param_count(self) -> tuple[int, int]:
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        return actor_params, actor_params + critic_params

    def save(self, checkpoint_path):
        if self.eval_mode: return
        if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, "PI"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, "V"))
        if self.actor_logstd is not None:
            torch.save(self.actor_logstd, os.path.join(checkpoint_path, "actor_logstd"))

    def load(self, checkpoint_path):
        self._get_torch_params(self.encoder)
        self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, "PI")))
        self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, "V")))
        if self.actor_logstd is not None:
            self.actor_logstd = torch.load(os.path.join(checkpoint_path, "actor_logstd"))
