from flexibuddiesrl.Agent import Agent, StochasticActor, ValueS, QS
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
import copy


class SAC(Agent):

    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        discrete_action_dims=None,
        max_actions=None,
        min_actions=None,
        hidden_dims=[32, 32],
        encoder=None,
        device="cpu",
        gumbel_tau=2.0,
        gumbel_tau_decay=0.9995,
        gumbel_tau_min=0.01,
        gumbel_hard=True,
        orthogonal_init=True,
        activation="tanh",
        action_head_hidden_dims=None,
        log_std_clamp_range=[-5, 1],
        lr=1e-3,
        actor_ratio=0.5,
        actor_every=1,
        gamma=0.99,
        sac_tau=0.05,
        initial_temperature=0.2,
        mode="V",  # V or Q
    ):
        assert mode in [
            "Q",
            "V",
        ], f"The critic mode needs to be 'V' or 'Q', you entered {mode}"
        if discrete_action_dims is None:
            self.critic_mode = "V"
            mode = "V"
        self.critic_mode = mode
        self.log_std_clamp_range = log_std_clamp_range
        self.actor_every = actor_every
        self.actor_ratio = actor_ratio
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.orthogonal_init = orthogonal_init
        action_dim = continuous_action_dim
        if discrete_action_dims is not None:
            action_dim += sum(discrete_action_dims)
        self.lr = lr
        self.gamma = gamma
        self.sac_tau = sac_tau
        self.initial_temperature = initial_temperature
        self.device = device
        self.obs_dim = obs_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = (
            list(discrete_action_dims) if discrete_action_dims is not None else None
        )
        self.action_dim = action_dim
        self.max_actions = (
            torch.as_tensor(max_actions, device=device, dtype=torch.float32)
            if max_actions is not None
            else None
        )
        self.min_actions = (
            torch.as_tensor(min_actions, device=device, dtype=torch.float32)
            if min_actions is not None
            else None
        )

        self.has_continuous = False
        if self.continuous_action_dim > 0:
            self.has_continuous = True
        self.has_discrete = False
        if self.discrete_action_dims is not None:
            self.has_discrete = True

        # Debug prints removed to avoid blocking input during training
        self.actor = StochasticActor(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            hidden_dims=hidden_dims,
            encoder=encoder,
            device=device,
            gumbel_tau=gumbel_tau,
            gumbel_tau_decay=gumbel_tau_decay,
            gumbel_tau_min=gumbel_tau_min,
            gumbel_hard=gumbel_hard,
            orthogonal_init=orthogonal_init,
            activation=activation,
            action_head_hidden_dims=action_head_hidden_dims,
            log_std_clamp_range=log_std_clamp_range,
            std_type="full",
            clamp_type="tanh",
        )
        self.encoder = self.actor.encoder
        # print(f" encoder in sac init: {self.encoder}")

        # print(
        #     f"obs dim: {obs_dim + action_dim} hdden dims: {hidden_dims} device: {device}"
        # )
        # input()
        # input()
        if self.critic_mode == "V":
            self._get_V_critics()
        else:
            self._get_Q_critics()
        # Set up optimizers (actor, critic, temperature)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr * actor_ratio
        )
        self.Q1_opt = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.Q2_opt = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)

        # Dual temperature: separate α_c (continuous) and α_d (discrete)
        _log_t0 = float(np.log(self.initial_temperature))
        self.log_alpha_c = torch.nn.Parameter(
            torch.tensor(_log_t0, device=self.device, dtype=torch.float32)
        )
        self.log_alpha_d = torch.nn.Parameter(
            torch.tensor(_log_t0, device=self.device, dtype=torch.float32)
        )
        self.alpha_opt_c = torch.optim.Adam([self.log_alpha_c], lr=self.lr)
        self.alpha_opt_d = torch.optim.Adam([self.log_alpha_d], lr=self.lr)

        # Separate target entropies
        self.target_entropy_c = (
            -float(self.continuous_action_dim) if self.has_continuous else 0.0
        )
        self.target_entropy_d = (
            -float(np.sum(np.log(self.discrete_action_dims)))
            if self.has_discrete
            else 0.0
        )
        # if self.critic_mode == "V":
        #     print(f"Q1 dtype: {self.Q1.l1.weight.dtype} device: {self.Q1.fc1.weight.device}")
        #     print(f"Q2 dtype: {self.Q2.l1.weight.dtype} device: {self.Q2.fc1.weight.device}")
        #     print(f"Target_Q1 dtype: {self.Q1_target.l1.weight.dtype} device: {self.Q1_target.fc1.weight.device}")
        #     print(f"Target_Q2 dtype: {self.Q2_target.l1.weight.dtype} device: {self.Q2_target.fc1.weight.device}")

        # else:
        #     for p in self.Q1.parameters():
        #         print(f"  Q1 param dtype: {p.dtype} device: {p.device}")
        #     for p in self.Q2.parameters():
        #         print(f"  Q2 param dtype: {p.dtype} device: {p.device}")
        #     for p in self.Q1_target.parameters():
        #         print(f"  Target Q1 param dtype: {p.dtype} device: {p.device}")
        #     for p in self.Q2_target.parameters():
        #         print(f"  Target Q2 param dtype: {p.dtype} device: {p.device}")
        # for a in self.actor.action_layers:
        #     for p in a.parameters():
        #         print(f"  Actor action head param dtype: {p.dtype} device: {p.device}")
        # print(f"encoder dtype: {self.encoder.encoder[0].weight.dtype} device: {self.encoder.encoder[0].weight.device}")
        # input()
        # Update cadence control
        self._step_counter = 0

    def _get_V_critics(self):
        self.Q1 = ValueS(
            obs_dim=self.obs_dim + self.action_dim,
            hidden_dim=self.hidden_dims[0],
            device=self.device,
            activation=self.activation,
            orthogonal_init=self.orthogonal_init,
        )
        self.Q2 = ValueS(
            obs_dim=self.obs_dim + self.action_dim,
            hidden_dim=self.hidden_dims[0],
            device=self.device,
            activation=self.activation,
            orthogonal_init=self.orthogonal_init,
        )

        self.Q1_target = ValueS(
            obs_dim=self.obs_dim + self.action_dim,
            hidden_dim=self.hidden_dims[0],
            device=self.device,
            activation=self.activation,
            orthogonal_init=self.orthogonal_init,
        )
        self.Q2_target = ValueS(
            obs_dim=self.obs_dim + self.action_dim,
            hidden_dim=self.hidden_dims[0],
            device=self.device,
            activation=self.activation,
            orthogonal_init=self.orthogonal_init,
        )
        for p in self.Q1_target.parameters():
            p.requires_grad_(False)
        for p in self.Q2_target.parameters():
            p.requires_grad_(False)
        self.Q1_target.eval()
        self.Q2_target.eval()

        # Hard copy critic->target initially
        self._hard_update(self.Q1_target, self.Q1)
        self._hard_update(self.Q2_target, self.Q2)

    def _get_Q_critics(self):
        self.Q1 = QS(
            self.obs_dim + self.continuous_action_dim,
            continuous_action_dim=0,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=self.hidden_dims,
            encoder=None,
            activation=self.activation,
            orthogonal=self.orthogonal_init,
            dropout=0.0,
            dueling=True,
            device=self.device,
            n_c_action_bins=0,
            head_hidden_dims=None,  # linear action heads
            verbose=False,
            QMIX=False,
        )
        self.Q2 = QS(
            self.obs_dim + self.continuous_action_dim,
            continuous_action_dim=0,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=self.hidden_dims,
            encoder=None,
            activation=self.activation,
            orthogonal=self.orthogonal_init,
            dropout=0.0,
            dueling=True,
            device=self.device,
            n_c_action_bins=0,
            head_hidden_dims=None,  # linear action heads
            verbose=False,
            QMIX=False,
        )

        self.Q1_target = QS(
            self.obs_dim + self.continuous_action_dim,
            continuous_action_dim=0,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=self.hidden_dims,
            encoder=None,
            activation=self.activation,
            orthogonal=self.orthogonal_init,
            dropout=0.0,
            dueling=True,
            device=self.device,
            n_c_action_bins=0,
            head_hidden_dims=None,  # linear action heads
            verbose=False,
            QMIX=False,
        )
        self.Q2_target = QS(
            self.obs_dim + self.continuous_action_dim,
            continuous_action_dim=0,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=self.hidden_dims,
            encoder=None,
            activation=self.activation,
            orthogonal=self.orthogonal_init,
            dropout=0.0,
            dueling=True,
            device=self.device,
            n_c_action_bins=0,
            head_hidden_dims=None,  # linear action heads
            verbose=False,
            QMIX=False,
        )
        for p in self.Q1_target.parameters():
            p.requires_grad = False
        for p in self.Q2_target.parameters():
            p.requires_grad = False
        self.Q1_target.eval()
        self.Q2_target.eval()
        # Hard copy critic->target initially
        self._hard_update(self.Q1_target, self.Q1)
        self._hard_update(self.Q2_target, self.Q2)

    def tonumpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to("cpu").numpy()
        return x

    def train_actions(
        self, observations, action_mask=None, step=False, debug=False
    ) -> dict:

        with torch.no_grad():
            continuous_means, continuous_log_std_logits, discrete_logits = self.actor(
                torch.tensor(observations, device=self.device),
                action_mask=action_mask,
                debug=debug,
            )
            (
                discrete_actions,
                continuous_actions,
                discrete_log_probs,
                continuous_log_probs,
                continuous_activations,
            ) = self.actor.action_from_logits(
                continuous_means,
                continuous_log_std_logits,
                discrete_logits,
                gumbel=False,  # This will be true in reinforcement learn to get a gradient
                log_con=False,
                log_disc=False,
            )

        return {
            "discrete_actions": self.tonumpy(discrete_actions),
            "continuous_actions": self.tonumpy(continuous_actions),
        }

    def ego_actions(self, observations, action_mask=None) -> dict:
        # Deterministic/selective policy: argmax over discrete heads and mean for continuous
        with torch.no_grad():
            obs_t = self._to_tensor(observations, torch.float32)
            c_means, c_logstd_logits, d_logits = self.actor(
                obs_t, action_mask=action_mask, debug=False
            )
            # Discrete: argmax per head
            if d_logits is not None:
                d_actions_idx = [torch.argmax(logit, dim=-1) for logit in d_logits]
            else:
                d_actions_idx = None
            # Continuous: clamp via tanh to [-1,1], then scale if bounds provided
            if c_means is not None:
                c_act = torch.tanh(c_means)
                if self.max_actions is not None and self.min_actions is not None:
                    c_act = self._scale_to_bounds(c_act)
            else:
                c_act = None
        return {"discrete_action": d_actions_idx, "continuous_action": c_act}

    def imitation_learn(
        self,
        observations,
        continuous_actions,
        discrete_actions,
        action_mask=None,
        debug=False,
    ) -> dict:
        # Note: Full continuous MLE under tanh-squashed Gaussian is non-trivial.
        # Here we implement discrete-behavior cloning if discrete actions exist.
        obs_t = self._to_tensor(observations, torch.float32)
        d_actor_loss = torch.tensor(0.0, device=self.device)
        c_actor_loss = torch.tensor(0.0, device=self.device)

        c_means, c_logstd_logits, d_logits = self.actor(
            obs_t, action_mask=action_mask, debug=debug
        )

        # Discrete MLE
        if (
            self.discrete_action_dims is not None
            and d_logits is not None
            and discrete_actions is not None
        ):
            if isinstance(discrete_actions, (list, tuple)):
                gt_disc = [self._to_tensor(a, torch.long) for a in discrete_actions]
            else:
                # assume shape [B, n_heads]
                da = self._to_tensor(discrete_actions, torch.long)
                gt_disc = [da[:, i] for i in range(da.shape[1])]
            ce_losses = [
                F.cross_entropy(logit, idx) for logit, idx in zip(d_logits, gt_disc)
            ]
            d_actor_loss = torch.stack(ce_losses).mean()

        # Optional: continuous MLE omitted (requires tanh-Gaussian correction)

        total_actor_loss = d_actor_loss + c_actor_loss
        if total_actor_loss.requires_grad and total_actor_loss > 0:
            self.actor_opt.zero_grad(set_to_none=True)
            total_actor_loss.backward()
            self.actor_opt.step()

        immitation_metrics = {
            "critic_loss": float(0.0),
            "actor_loss": float(total_actor_loss.detach().item()),
            "time": 0,
        }
        return immitation_metrics

    def utility_function(self, observations, actions=None):
        # Returns Q(s,a) if actions provided; else returns V(s)=E_a[Q(s,a)-alpha*logpi]
        obs_t = self._to_tensor(observations, torch.float32)
        if actions is not None:
            a_vec = self._build_action_vector_from_actions(actions)
            q_in = torch.cat([obs_t, a_vec], dim=-1)
            q = self.Q1(q_in)
            return q
        # V(s) with gradient attached
        alpha_c = self._alpha_c()
        alpha_d = self._alpha_d()
        c_means, c_logstd_logits, d_logits = self.actor(
            obs_t, action_mask=None, debug=False
        )
        (
            d_act,
            c_act,
            d_logp,
            c_logp,
            _,
        ) = self.actor.action_from_logits(
            c_means,
            c_logstd_logits,
            d_logits,
            gumbel=True,
            log_con=self.has_continuous,
            log_disc=self.has_discrete,
        )
        a_vec = self._flatten_actions(c_act, d_act)
        q_in = torch.cat([obs_t, a_vec], dim=-1)
        q = self.Q1(q_in)
        q = q.squeeze(-1) if q.ndim > 1 else q
        c_lp = self._aggregate_continuous_logp(c_logp)
        ent_pen = obs_t.new_zeros(obs_t.shape[0])
        if c_lp is not None:
            ent_pen = ent_pen + alpha_c * c_lp
        if d_logits is not None and len(d_logits) > 0:
            d_neg_ent = self._discrete_neg_entropy(d_logits, detach=False)
            ent_pen = ent_pen + alpha_d * d_neg_ent
        return q - ent_pen

    def expected_V(self, obs, legal_action) -> torch.Tensor | np.ndarray | float:
        # V(s) with no gradient
        with torch.no_grad():
            obs_t = self._to_tensor(obs, torch.float32)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            alpha_c = self._alpha_c()
            alpha_d = self._alpha_d()
            c_means, c_logstd_logits, d_logits = self.actor(
                obs_t, action_mask=None, debug=False
            )
            # Sample continuous action; for discrete we compute expectation
            (
                d_act,
                c_act,
                d_logp,
                c_logp,
                _,
            ) = self.actor.action_from_logits(
                c_means,
                c_logstd_logits,
                d_logits,
                gumbel=True,
                log_con=self.has_continuous,
                log_disc=(self.critic_mode == "V" and self.has_discrete),
            )
            if self.critic_mode == "V":
                a_vec = self._flatten_actions(c_act, d_act)
                q_in = torch.cat([obs_t, a_vec], dim=-1)
                q = self.Q1_target(q_in)
                q = q.squeeze(-1) if q.ndim > 1 else q
                ent_pen = obs_t.new_zeros(obs_t.shape[0])
                c_lp = self._aggregate_continuous_logp(c_logp)
                if c_lp is not None:
                    ent_pen = ent_pen + alpha_c * c_lp
                if d_logits is not None and len(d_logits) > 0:
                    d_neg_ent = self._discrete_neg_entropy(d_logits)
                    ent_pen = ent_pen + alpha_d * d_neg_ent
                return q - ent_pen
            else:
                # Q-mode: soft V via logsumexp over advantage heads
                if c_act is not None and isinstance(c_act, (list, tuple)):
                    c_act_cat = torch.cat(c_act, dim=-1)
                else:
                    c_act_cat = c_act
                in_vec = (
                    torch.cat([obs_t, c_act_cat], dim=-1)
                    if c_act_cat is not None
                    else obs_t
                )
                v1, adv1, _ = self.Q1_target(in_vec)
                v2, adv2, _ = self.Q2_target(in_vec)
                V_soft_1 = self._soft_v(v1, adv1, alpha_d)
                V_soft_2 = self._soft_v(v2, adv2, alpha_d)
                V_soft_min = torch.minimum(V_soft_1, V_soft_2)
                c_logp_agg = self._aggregate_continuous_logp(c_logp)
                if c_logp_agg is None:
                    c_logp_agg = torch.zeros_like(V_soft_min)
                v = V_soft_min - alpha_c * c_logp_agg
                return v

    def stable_greedy(self, obs, legal_action):
        """
        Sample a greedy action from this agent's target or stable
        policy. For DQN this is argmax(target_Q), for PPO/SAC this is
        just like taking a train action which is equal in
        expectation to the current policy.
        """
        with torch.no_grad():
            # Deterministic greedy action from the current policy
            out = self.ego_actions(obs)
            ca = out.get("continuous_action")
            if isinstance(ca, list):
                ca = ca[0]
            da = out.get("discrete_action")
            if isinstance(da, list):
                da = torch.stack(da, dim=-1)
            da = torch.as_tensor(da, device=self.device) if da is not None else None
            ca = torch.as_tensor(ca, device=self.device) if ca is not None else None
        return da, ca

    def reinforcement_learn(
        self, batch, agent_num=0, critic_only=False, debug=False
    ) -> dict:
        obs = batch.__getattr__("obs")[agent_num]
        obs_ = batch.__getattr__("obs_")[agent_num]
        rewards = batch.__getattr__("global_rewards")
        discrete_actions = batch.__getattr__("discrete_actions")[agent_num]
        continuous_actions = batch.__getattr__("continuous_actions")[agent_num]

        if obs.requires_grad:
            obs = obs.detach()
        if obs_.requires_grad:
            obs_ = obs_.detach()
        if rewards.requires_grad:
            rewards = rewards.detach()
        if discrete_actions is not None and discrete_actions.requires_grad:
            discrete_actions = discrete_actions.detach()
        if continuous_actions is not None and continuous_actions.requires_grad:
            continuous_actions = continuous_actions.detach()

        # Optional fields
        try:
            dones = batch.__getattr__("terminated")
        except Exception:
            try:
                dones = batch.__getattr__("done")
            except Exception:
                dones = None

        obs_t = self._to_tensor(obs, torch.float32)
        obs_next_t = self._to_tensor(obs_, torch.float32)
        r_t = self._to_tensor(rewards, torch.float32).squeeze(-1)
        if dones is not None:
            d_t = self._to_tensor(dones, torch.float32).squeeze(-1)
        else:
            d_t = torch.zeros_like(r_t)

        alpha_c = self._alpha_c()
        alpha_d = self._alpha_d()
        is_v = self.critic_mode == "V"

        # Build next-state target V
        with torch.no_grad():
            c_means_n, c_logstd_logits_n, d_logits_n = self.actor(
                obs_next_t, action_mask=None, debug=debug
            )
            (
                d_act_n,
                c_act_n,
                d_logp_n,
                c_logp_n,
                _,
            ) = self.actor.action_from_logits(
                c_means_n,
                c_logstd_logits_n,
                d_logits_n,
                gumbel=True,
                log_con=self.has_continuous,
                log_disc=(is_v and self.has_discrete),
            )
            if is_v:
                a_next_vec = self._flatten_actions(c_act_n, d_act_n)
                q_target_in = torch.cat([obs_next_t, a_next_vec], dim=-1)
                q_next = torch.minimum(
                    self.Q1_target(q_target_in), self.Q2_target(q_target_in)
                )
                q_next = q_next.squeeze(-1) if q_next.ndim > 1 else q_next
                ent_pen_n = obs_next_t.new_zeros(obs_next_t.shape[0])
                c_lp_n = self._aggregate_continuous_logp(c_logp_n)
                if c_lp_n is not None:
                    ent_pen_n = ent_pen_n + alpha_c * c_lp_n
                if d_logits_n is not None and len(d_logits_n) > 0:
                    d_neg_ent_n = self._discrete_neg_entropy(d_logits_n)
                    ent_pen_n = ent_pen_n + alpha_d * d_neg_ent_n
                v_next = q_next - ent_pen_n
            else:
                # Q-mode: soft V via logsumexp over discrete advantage heads
                # V_soft(s',a_c') = V(s',a_c') + Σ_i α_d·logsumexp(A_i(s',a_c',·)/α_d)
                if c_act_n is not None and isinstance(c_act_n, (list, tuple)):
                    c_act_n_cat = torch.cat(c_act_n, dim=-1)
                else:
                    c_act_n_cat = c_act_n
                in_vec_next = (
                    torch.cat([obs_next_t, c_act_n_cat], dim=-1)
                    if c_act_n_cat is not None
                    else obs_next_t
                )
                v1_n, adv1_n, _ = self.Q1_target(in_vec_next)
                v2_n, adv2_n, _ = self.Q2_target(in_vec_next)
                V_soft_min_n = torch.minimum(
                    self._soft_v(v1_n, adv1_n, alpha_d),
                    self._soft_v(v2_n, adv2_n, alpha_d),
                )
                c_logp_n_agg = self._aggregate_continuous_logp(c_logp_n)
                if c_logp_n_agg is None:
                    c_logp_n_agg = V_soft_min_n.new_zeros(V_soft_min_n.shape[0])
                v_next = V_soft_min_n - alpha_c * c_logp_n_agg
            y = r_t + (1.0 - d_t) * (self.gamma * v_next)

        # Current Q(s,a)
        if is_v:
            a_vec_v = self._build_action_vector_from_batch_actions(
                continuous_actions, discrete_actions
            )
            q_in = torch.cat([obs_t, a_vec_v], dim=-1)
            q_1 = self.Q1(q_in)
            q_1 = q_1.squeeze(-1) if q_1.ndim > 1 else q_1
            q_2 = self.Q2(q_in)
            q_2 = q_2.squeeze(-1) if q_2.ndim > 1 else q_2
        else:
            c_hist = (
                self._to_tensor(continuous_actions, torch.float32)
                if self.has_continuous
                else None
            )
            in_vec_cur = (
                torch.cat([obs_t, c_hist], dim=-1) if c_hist is not None else obs_t
            )
            v1, adv1, _ = self.Q1(in_vec_cur)
            v2, adv2, _ = self.Q2(in_vec_cur)
            adv1_sum = self._gather_sum_adv(adv1, discrete_actions)
            adv2_sum = self._gather_sum_adv(adv2, discrete_actions)
            q_1 = v1.squeeze(-1) + adv1_sum
            q_2 = v2.squeeze(-1) + adv2_sum

        critic_loss_1 = F.mse_loss(q_1, y)
        critic_loss_2 = F.mse_loss(q_2, y)

        # Combined backward: Q1 and Q2 share no parameters
        self.Q1_opt.zero_grad(set_to_none=True)
        self.Q2_opt.zero_grad(set_to_none=True)
        (critic_loss_1 + critic_loss_2).backward()
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0)
        self.Q1_opt.step()
        self.Q2_opt.step()

        self._soft_update(self.Q1_target, self.Q1, self.sac_tau)
        self._soft_update(self.Q2_target, self.Q2, self.sac_tau)

        # Actor/temperature updates
        d_actor_loss_val = 0.0
        c_actor_loss_val = 0.0
        d_entropy_val = 0.0
        c_entropy_val = 0.0
        c_std_mean_val = 0.0

        self._step_counter += 1
        update_actor = self._step_counter % self.actor_every == 0

        if update_actor:
            c_means, c_logstd_logits, d_logits = self.actor(
                obs_t, action_mask=None, debug=debug
            )
            (
                d_act_s,
                c_act_s,
                d_logp_s,
                c_logp_s,
                _,
            ) = self.actor.action_from_logits(
                c_means,
                c_logstd_logits,
                d_logits,
                gumbel=True,
                log_con=self.has_continuous,
                log_disc=(is_v and self.has_discrete),
            )

            # Pre-aggregate continuous log probs once (reused for loss + metrics)
            c_logp_agg = self._aggregate_continuous_logp(c_logp_s)

            c_logp_for_alpha = None
            d_logp_for_alpha = None

            if is_v:
                a_sample_vec = self._flatten_actions(c_act_s, d_act_s)
                q_pi_in = torch.cat([obs_t, a_sample_vec], dim=-1)
                q_pi = torch.minimum(self.Q1(q_pi_in), self.Q2(q_pi_in))
                q_pi = q_pi.squeeze(-1) if q_pi.ndim > 1 else q_pi

                # Separate entropy penalties per action space
                ent_pen = obs_t.new_zeros(obs_t.shape[0])
                if c_logp_agg is not None:
                    ent_pen = ent_pen + alpha_c * c_logp_agg
                if d_logits is not None and len(d_logits) > 0:
                    d_neg_ent = self._discrete_neg_entropy(d_logits, detach=False)
                    ent_pen = ent_pen + alpha_d * d_neg_ent

                actor_loss = (ent_pen - q_pi).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_opt.step()

                # Detached logps for separate alpha updates
                if c_logp_agg is not None:
                    c_logp_for_alpha = c_logp_agg.detach()
                if d_logits is not None and len(d_logits) > 0:
                    d_logp_for_alpha = d_neg_ent.detach()

                # V-mode metrics
                q_pi_mean_det = q_pi.detach().mean()
                if d_logits is not None and len(d_logits) > 0:
                    d_neg_ent = self._discrete_neg_entropy(d_logits)
                    d_entropy_val = float((-d_neg_ent).mean().item())
                    d_actor_loss_val = float(
                        (alpha_d.detach() * d_neg_ent.mean() - q_pi_mean_det).item()
                    )
                if c_logp_agg is not None:
                    c_actor_loss_val = float(
                        (
                            alpha_c.detach() * c_logp_agg.detach().mean()
                            - q_pi_mean_det
                        ).item()
                    )

            else:
                # ---- Q-mode actor update ----
                # Continuous: maximise V_soft(s, a_c) - α_c·log π_c(a_c|s)
                # Discrete:  minimise KL(π_θ || softmax(Q_min / α_d))
                c_act_s_cat = (
                    torch.cat(c_act_s, dim=-1)
                    if isinstance(c_act_s, (list, tuple))
                    else c_act_s
                )
                in_vec = (
                    torch.cat([obs_t, c_act_s_cat], dim=-1)
                    if c_act_s_cat is not None
                    else obs_t
                )
                v1_pi, adv1_pi, _ = self.Q1(in_vec)
                v2_pi, adv2_pi, _ = self.Q2(in_vec)

                # Continuous loss: V_soft = V + Σ_i α_d·logsumexp(A_i/α_d)
                V_soft_min = torch.minimum(
                    self._soft_v(v1_pi, adv1_pi, alpha_d),
                    self._soft_v(v2_pi, adv2_pi, alpha_d),
                )
                c_logp_safe = (
                    c_logp_agg
                    if c_logp_agg is not None
                    else V_soft_min.new_zeros(V_soft_min.shape[0])
                )
                c_actor_loss = (alpha_c * c_logp_safe - V_soft_min).mean()

                # Discrete KL + accumulate detached neg-entropy for alpha & metrics
                d_kl_loss = obs_t.new_zeros(())
                d_neg_ent_sum = obs_t.new_zeros(obs_t.shape[0])  # [B]
                if d_logits is not None and adv1_pi is not None:
                    alpha_d_det = alpha_d.detach()
                    for i, logit_i in enumerate(d_logits):
                        log_pi_i = F.log_softmax(logit_i, dim=-1)
                        pi_i = log_pi_i.exp()
                        q_min_i = torch.minimum(v1_pi + adv1_pi[i], v2_pi + adv2_pi[i])
                        log_target_pi_i = F.log_softmax(
                            q_min_i / alpha_d_det, dim=-1
                        ).detach()
                        d_kl_loss = (
                            d_kl_loss
                            + (pi_i * (log_pi_i - log_target_pi_i)).sum(dim=-1).mean()
                        )
                        # Cache detached neg-entropy (avoids 2 extra softmax passes)
                        d_neg_ent_sum = d_neg_ent_sum + (
                            pi_i.detach() * log_pi_i.detach()
                        ).sum(dim=-1)

                actor_loss = c_actor_loss + d_kl_loss
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_opt.step()

                # Separate logps for dual alpha updates
                if c_logp_agg is not None:
                    c_logp_for_alpha = c_logp_agg.detach()
                if self.has_discrete:
                    d_logp_for_alpha = d_neg_ent_sum.detach()

                # Q-mode metrics
                d_entropy_val = float((-d_neg_ent_sum).mean().item())
                d_actor_loss_val = float(d_kl_loss.detach().item())
                c_actor_loss_val = float(c_actor_loss.detach().item())

            # --- Dual temperature updates ---
            if c_logp_for_alpha is not None:
                alpha_c_loss = -(
                    self.log_alpha_c * (c_logp_for_alpha.mean() + self.target_entropy_c)
                )
                self.alpha_opt_c.zero_grad(set_to_none=True)
                alpha_c_loss.backward()
                self.alpha_opt_c.step()
            if d_logp_for_alpha is not None:
                alpha_d_loss = -(
                    self.log_alpha_d * (d_logp_for_alpha.mean() + self.target_entropy_d)
                )
                self.alpha_opt_d.zero_grad(set_to_none=True)
                alpha_d_loss.backward()
                self.alpha_opt_d.step()

            # Common metrics: continuous entropy and std
            if c_logp_agg is not None:
                c_entropy_val = float((-c_logp_agg.detach()).mean().item())
            if c_logstd_logits is not None:
                c_std_mean_val = float(
                    c_logstd_logits.detach()
                    .clamp(self.log_std_clamp_range[0], self.log_std_clamp_range[1])
                    .exp()
                    .mean()
                    .item()
                )

        return {
            "critic_loss": 0.5 * float(critic_loss_1.item() + critic_loss_2.item()),
            "d_actor_loss": d_actor_loss_val,
            "c_actor_loss": c_actor_loss_val,
            "d_entropy": d_entropy_val,
            "c_entropy": c_entropy_val,
            "c_std": c_std_mean_val,
        }

    def save(self, checkpoint_path):
        # Save the model in the checkpoint path
        state = {
            "actor": self.actor.state_dict(),
            "critic_1": self.Q1.state_dict(),
            "critic_target_1": self.Q1_target.state_dict(),
            "critic_2": self.Q2.state_dict(),
            "critic_target_2": self.Q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt_1": self.Q1_opt.state_dict(),
            "critic_opt_2": self.Q2_opt.state_dict(),
            "alpha_opt_c": self.alpha_opt_c.state_dict(),
            "alpha_opt_d": self.alpha_opt_d.state_dict(),
            "log_alpha_c": self.log_alpha_c.detach().cpu(),
            "log_alpha_d": self.log_alpha_d.detach().cpu(),
            "config": {
                "obs_dim": self.obs_dim,
                "continuous_action_dim": self.continuous_action_dim,
                "discrete_action_dims": self.discrete_action_dims,
            },
        }
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        # Save the model from the checkpoint path
        chkpt = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(chkpt["actor"])
        self.Q1.load_state_dict(chkpt["critic_1"])
        self.Q1_target.load_state_dict(chkpt["critic_target_1"])
        self.Q2.load_state_dict(chkpt["critic_2"])
        self.Q2_target.load_state_dict(chkpt["critic_target_2"])
        self.actor_opt.load_state_dict(chkpt["actor_opt"])
        self.Q1_opt.load_state_dict(chkpt["critic_opt_1"])
        self.Q2_opt.load_state_dict(chkpt["critic_opt_2"])
        # Dual alpha: load new keys if present, fall back to legacy single alpha
        if "alpha_opt_c" in chkpt:
            self.alpha_opt_c.load_state_dict(chkpt["alpha_opt_c"])
            self.alpha_opt_d.load_state_dict(chkpt["alpha_opt_d"])
        elif "alpha_opt" in chkpt:
            self.alpha_opt_c.load_state_dict(chkpt["alpha_opt"])
            self.alpha_opt_d.load_state_dict(chkpt["alpha_opt"])
        if "log_alpha_c" in chkpt:
            with torch.no_grad():
                self.log_alpha_c.copy_(chkpt["log_alpha_c"].to(self.device).float())
                self.log_alpha_d.copy_(chkpt["log_alpha_d"].to(self.device).float())
        elif "log_alpha" in chkpt:
            with torch.no_grad():
                self.log_alpha_c.copy_(chkpt["log_alpha"].to(self.device).float())
                self.log_alpha_d.copy_(chkpt["log_alpha"].to(self.device).float())

    def param_count(self) -> tuple[int, int]:
        # First number is the policy param count
        # Second is the critic + policy param counts
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.Q1.parameters())
        return (
            actor_params,
            actor_params + critic_params,
        )  # train and execute param count

    # -------------------------
    # Helpers
    # -------------------------
    def _hard_update(self, target: torch.nn.Module, source: torch.nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(
        self, target: torch.nn.Module, source: torch.nn.Module, tau: float
    ):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.lerp_(sp.data, tau)

    def _alpha_c(self) -> torch.Tensor:
        return self.log_alpha_c.exp()

    def _alpha_d(self) -> torch.Tensor:
        return self.log_alpha_d.exp()

    def _to_tensor(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _scale_to_bounds(self, c_act: torch.Tensor) -> torch.Tensor:
        # Scale from [-1,1] to [min,max]
        if self.max_actions is None or self.min_actions is None:
            return c_act
        center = (self.max_actions + self.min_actions) / 2.0
        half_range = (self.max_actions - self.min_actions) / 2.0
        return center + half_range * c_act

    def _one_hot_discrete(self, d_actions: List[torch.Tensor]) -> List[torch.Tensor]:
        assert self.discrete_action_dims is not None
        one_hots: List[torch.Tensor] = []
        for idx, a_idx in enumerate(d_actions):
            a_idx = a_idx.to(self.device)
            oh = F.one_hot(
                a_idx.long(), num_classes=int(self.discrete_action_dims[idx])
            ).float()
            one_hots.append(oh)
        return one_hots

    def _flatten_actions(
        self,
        c_act: Optional[torch.Tensor | List[torch.Tensor]],
        d_act: Optional[List[torch.Tensor] | torch.Tensor],
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if c_act is not None:
            if isinstance(c_act, (list, tuple)):
                parts.extend([ca if ca.ndim > 1 else ca.unsqueeze(0) for ca in c_act])
            else:
                parts.append(c_act if c_act.ndim > 1 else c_act.unsqueeze(0))
        if d_act is not None:
            if isinstance(d_act, (list, tuple)):
                parts.extend(d_act)
            else:
                # If a single tensor for discrete was provided, assume already one-hot flattened
                parts.append(d_act)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _build_action_vector_from_batch_actions(
        self, c_actions: Any, d_actions: Any
    ) -> torch.Tensor:
        a_vec = []
        if self.has_continuous:
            a_vec.append(c_actions)
        if self.has_discrete:
            for i, d in enumerate(self.discrete_action_dims):  # type:ignore
                a_vec.append(F.one_hot(d_actions[:, i], d))
        return torch.cat(a_vec, dim=-1)

    def _build_action_vector_from_actions(
        self, actions: Dict[str, Any]
    ) -> torch.Tensor:
        c_act = actions.get("continuous_action")
        d_act = actions.get("discrete_action")
        c_act_t = self._to_tensor(c_act, torch.float32) if c_act is not None else None
        if isinstance(d_act, (list, tuple)):
            d_idxs = [self._to_tensor(a, torch.long) for a in d_act]
        elif d_act is not None:
            d_t = self._to_tensor(d_act, torch.long)
            if d_t.ndim == 2:
                d_idxs = [d_t[:, i] for i in range(d_t.shape[1])]
            else:
                d_idxs = [d_t]
        else:
            d_idxs = None
        d_oh = self._one_hot_discrete(d_idxs) if d_idxs is not None else None
        return self._flatten_actions(c_act_t, d_oh)

    def _sum_logps(
        self,
        c_logp: Optional[torch.Tensor | List[torch.Tensor]],
        d_logp: Optional[torch.Tensor | List[torch.Tensor]],
    ) -> torch.Tensor:
        def cont_sum(
            x: Optional[torch.Tensor | List[torch.Tensor]],
        ) -> Optional[torch.Tensor]:
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                # Sum across provided continuous parts (rare), then across dims
                xs = [t if t.ndim == 1 else t.sum(dim=-1) for t in x]
                out = xs[0]
                for t in xs[1:]:
                    out = out + t
                return out
            return x.sum(dim=-1) if x.ndim > 1 else x

        def disc_sum(
            x: Optional[torch.Tensor | List[torch.Tensor]],
        ) -> Optional[torch.Tensor]:
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return torch.stack(
                    [t if t.ndim == 1 else t.squeeze(-1) for t in x], dim=-1
                ).sum(dim=-1)
            return x if x.ndim == 1 else x.squeeze(-1)

        c_s = cont_sum(c_logp)
        d_s = disc_sum(d_logp)
        if c_s is not None and d_s is not None:
            return c_s + d_s
        if c_s is not None:
            return c_s
        if d_s is not None:
            return d_s
        # Fallback: zero
        return torch.tensor(0.0, device=self.device)

    def _aggregate_discrete_logp(
        self, d_logp: Optional[torch.Tensor | List[torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        if d_logp is None:
            return None
        if isinstance(d_logp, (list, tuple)):
            return torch.stack(
                [t if t.ndim == 1 else t.squeeze(-1) for t in d_logp], dim=-1
            ).sum(dim=-1)
        return d_logp if d_logp.ndim == 1 else d_logp.squeeze(-1)

    def _aggregate_continuous_logp(
        self, c_logp: Optional[torch.Tensor | List[torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        if c_logp is None:
            return None
        if isinstance(c_logp, (list, tuple)):
            xs = [t if t.ndim == 1 else t.sum(dim=-1) for t in c_logp]
            out = xs[0]
            for t in xs[1:]:
                out = out + t
            return out
        return c_logp.sum(dim=-1) if c_logp.ndim > 1 else c_logp

    # Expected advantage helper used in Q-mode
    def _exp_adv(
        self,
        adv_heads: Optional[List[torch.Tensor]],
        pi_heads: Optional[List[torch.Tensor]],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        # Returns a [B] tensor with expected advantage across all discrete heads.
        if adv_heads is None or pi_heads is None or len(pi_heads) == 0:
            return ref.new_zeros(ref.shape[0])
        out = ref.new_zeros(ref.shape[0])
        for i, adv in enumerate(adv_heads):
            out = out + (pi_heads[i] * adv).sum(dim=-1)
        return out

    def _soft_v(
        self,
        v: torch.Tensor,
        adv_heads: Optional[List[torch.Tensor]],
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft value: V_soft(s, a_c) = V(s,a_c) + Σ_i α · logsumexp(A_i(·)/α)
        This is the log-partition form of the soft Bellman equation for
        the discrete action dimensions, given the additive Q decomposition
        Q(s,a_c,a_d) = V(s,a_c) + Σ_i A_i(s,a_c,k_i).
        """
        sv = v.squeeze(-1)
        if adv_heads is not None:
            inv_alpha = 1.0 / alpha
            for adv in adv_heads:
                sv = sv + alpha * torch.logsumexp(adv * inv_alpha, dim=-1)
        return sv

    def _discrete_neg_entropy(
        self, d_logits: List[torch.Tensor], detach: bool = True
    ) -> torch.Tensor:
        """Compute Σ_i E_π[log π_i(k)] across discrete heads. Returns [B] tensor.
        When detach=False, gradients flow through for the actor loss."""
        out = d_logits[0].new_zeros(d_logits[0].shape[0])
        for lg in d_logits:
            logits = lg.detach() if detach else lg
            log_pi = F.log_softmax(logits, dim=-1)
            out = out + (log_pi.exp() * log_pi).sum(dim=-1)
        return out

    @staticmethod
    def _gather_sum_adv(
        adv_heads: List[torch.Tensor], d_idx: torch.Tensor
    ) -> torch.Tensor:
        """Sum gathered advantages across discrete heads for batch indices."""
        out = adv_heads[0].new_zeros(adv_heads[0].shape[0])
        for i, adv in enumerate(adv_heads):
            idx = d_idx[:, i].long().unsqueeze(-1)
            out = out + adv.gather(dim=-1, index=idx).squeeze(-1)
        return out
