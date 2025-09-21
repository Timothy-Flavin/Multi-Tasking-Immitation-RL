from flexibuddiesrl.Agent import Agent, StochasticActor, ValueS, QS
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional


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
        gumbel_tau=1.0,
        gumbel_tau_decay=0.9999,
        gumbel_tau_min=0.1,
        gumbel_hard=False,
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

        print(
            f"obs dim: {obs_dim + action_dim} hdden dims: {hidden_dims} device: {device}"
        )
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

        # Temperature parameter alpha with auto-entropy tuning
        # Use log_alpha for unconstrained optimization
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(
                np.log(self.initial_temperature),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # Target entropy: continuous dims use -dim, discrete dims use -sum(log(K))
        target_disc = 0.0
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            target_disc = float(np.sum(np.log(self.discrete_action_dims)))
        self.target_entropy = -float(self.continuous_action_dim) - target_disc

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
            "discrete_action": self.tonumpy(discrete_actions),
            "continuous_action": self.tonumpy(continuous_actions),
            "discrete_log_prob": self.tonumpy(discrete_log_probs),
            "continuous_log_prob": self.tonumpy(continuous_log_probs),
            "value": 0,
            "time": 0,
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
        alpha = self._alpha()
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
        logp_sum = self._sum_logps(c_logp, d_logp)
        return q - alpha * logp_sum

    def expected_V(self, obs, legal_action) -> torch.Tensor | np.ndarray | float:
        # V(s) with no gradient
        with torch.no_grad():
            obs_t = self._to_tensor(obs, torch.float32)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            alpha = self._alpha()
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
                logp_sum = self._sum_logps(c_logp, d_logp)
                v = q - alpha * logp_sum
                return v.squeeze(-1) if v.ndim > 1 else v
            else:
                # Q-mode: expectation over discrete heads
                # Ensure continuous action is a single tensor if present
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
                # Compute pi over discrete heads
                pi_list = (
                    [F.softmax(logits, dim=-1) for logits in d_logits]
                    if d_logits is not None
                    else []
                )

                # Expected advantages per head
                exp_adv1 = self._exp_adv(adv1, pi_list, v1)
                exp_adv2 = self._exp_adv(adv2, pi_list, v2)
                V1 = v1.squeeze(-1) + exp_adv1
                V2 = v2.squeeze(-1) + exp_adv2
                Vmin = torch.minimum(V1, V2)
                # E log pi over discrete heads
                logpi_d_exp = (
                    sum(
                        [
                            (pi * F.log_softmax(lg, dim=-1)).sum(dim=-1)
                            for pi, lg in zip(pi_list, d_logits)
                        ]
                    )
                    if d_logits is not None
                    else torch.zeros_like(Vmin)
                )
                c_logp_agg = self._aggregate_continuous_logp(c_logp)
                if c_logp_agg is None:
                    c_logp_agg = torch.zeros_like(Vmin)
                v = Vmin - alpha * (c_logp_agg + logpi_d_exp)
                return v

    def stable_greedy(self, obs, legal_action):
        """
        Sample a greedy action from this agent's target or stable
        policy. For DQN this is argmax(target_Q), for PPO/SAC this is
        just like taking a train action which is equal in
        expectation to the current policy.
        """
        # Deterministic greedy action from the current policy
        out = self.ego_actions(obs)
        return out.get("continuous_action"), out.get("discrete_action")

    def reinforcement_learn(
        self, batch, agent_num=0, critic_only=False, debug=False
    ) -> dict:
        obs = batch.__getattr__("obs")[agent_num]
        obs_ = batch.__getattr__("obs_")[agent_num]
        rewards = batch.__getattr__("global_reward")
        discrete_actions = batch.__getattr__("discrete_actions")[agent_num]
        continuous_actions = batch.__getattr__("continuous_actions")[agent_num]

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

        alpha = self._alpha()

        # Build next-state target V
        with torch.no_grad():
            # build current action vector for V-mode later when needed
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
                log_disc=(self.critic_mode == "V" and self.has_discrete),
            )
            if self.critic_mode == "V":
                a_next_vec = self._flatten_actions(c_act_n, d_act_n)
                q_target_in = torch.cat([obs_next_t, a_next_vec], dim=-1)
                q_next_1 = self.Q1_target(q_target_in)
                q_next_2 = self.Q2_target(q_target_in)
                q_next = torch.minimum(q_next_1, q_next_2)
                q_next = q_next.squeeze(-1) if q_next.ndim > 1 else q_next
                logp_next = self._sum_logps(c_logp_n, d_logp_n)
                v_next = q_next - alpha * logp_next
            else:
                # Q-mode: expectation over discrete heads given sampled continuous action
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
                pi_list_n = (
                    [F.softmax(lg, dim=-1) for lg in d_logits_n]
                    if d_logits_n is not None
                    else []
                )

                V1_n = v1_n.squeeze(-1) + self._exp_adv(adv1_n, pi_list_n, v1_n)
                V2_n = v2_n.squeeze(-1) + self._exp_adv(adv2_n, pi_list_n, v2_n)
                Vmin_n = torch.minimum(V1_n, V2_n)
                logpi_d_exp_n = (
                    sum(
                        [
                            (pi * F.log_softmax(lg, dim=-1)).sum(dim=-1)
                            for pi, lg in zip(pi_list_n, d_logits_n)
                        ]
                    )
                    if d_logits_n is not None
                    else Vmin_n.new_zeros(Vmin_n.shape[0])
                )
                c_logp_n_agg = self._aggregate_continuous_logp(c_logp_n)
                if c_logp_n_agg is None:
                    c_logp_n_agg = Vmin_n.new_zeros(Vmin_n.shape[0])
                v_next = Vmin_n - alpha * (c_logp_n_agg + logpi_d_exp_n)
            y = r_t + (1.0 - d_t) * (self.gamma * v_next)

        # Current Q(s,a)
        if self.critic_mode == "V":
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

            def gather_sum_adv(adv_heads, d_idx):
                out = v1.new_zeros(v1.shape[0])
                for i, adv in enumerate(adv_heads):
                    idx = d_idx[:, i].long().unsqueeze(-1)
                    out = out + adv.gather(dim=-1, index=idx).squeeze(-1)
                return out

            adv1_sum = gather_sum_adv(adv1, discrete_actions)
            adv2_sum = gather_sum_adv(adv2, discrete_actions)
            q_1 = v1.squeeze(-1) + adv1_sum
            q_2 = v2.squeeze(-1) + adv2_sum

        critic_loss_1 = F.mse_loss(q_1, y)
        critic_loss_2 = F.mse_loss(q_2, y)

        self.Q1_opt.zero_grad(set_to_none=True)
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=1.0)
        self.Q1_opt.step()

        self.Q2_opt.zero_grad(set_to_none=True)
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0)
        self.Q2_opt.step()

        self._soft_update(self.Q1_target, self.Q1, self.sac_tau)
        self._soft_update(self.Q2_target, self.Q2, self.sac_tau)

        # Actor/temperature updates
        d_actor_loss_val = torch.tensor(0.0, device=self.device)
        c_actor_loss_val = torch.tensor(0.0, device=self.device)
        d_entropy = torch.tensor(0.0, device=self.device)
        c_entropy = torch.tensor(0.0, device=self.device)
        c_std_mean = torch.tensor(0.0, device=self.device)

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
                log_disc=(self.critic_mode == "V" and self.has_discrete),
            )
            current_logp_for_alpha = None
            q_pi = None
            Vmin_pi = None
            if self.critic_mode == "V":
                a_sample_vec = self._flatten_actions(c_act_s, d_act_s)
                q_pi_in = torch.cat([obs_t, a_sample_vec], dim=-1)
                q_pi_1 = self.Q1(q_pi_in)
                q_pi_2 = self.Q2(q_pi_in)
                q_pi = torch.minimum(q_pi_1, q_pi_2)
                q_pi = q_pi.squeeze(-1) if q_pi.ndim > 1 else q_pi
                logp_s = self._sum_logps(c_logp_s, d_logp_s)
                current_logp_for_alpha = logp_s
                actor_loss = (alpha * logp_s - q_pi).mean()
            else:
                if c_act_s is not None and isinstance(c_act_s, (list, tuple)):
                    c_act_s_cat = torch.cat(c_act_s, dim=-1)
                else:
                    c_act_s_cat = c_act_s
                in_vec = (
                    torch.cat([obs_t, c_act_s_cat], dim=-1)
                    if c_act_s_cat is not None
                    else obs_t
                )
                v1, adv1, _ = self.Q1(in_vec)
                v2, adv2, _ = self.Q2(in_vec)
                pi_list = (
                    [F.softmax(logits, dim=-1) for logits in d_logits]
                    if d_logits is not None
                    else []
                )

                V1_pi = v1.squeeze(-1) + self._exp_adv(adv1, pi_list, v1)
                V2_pi = v2.squeeze(-1) + self._exp_adv(adv2, pi_list, v2)
                Vmin_pi = torch.minimum(V1_pi, V2_pi)
                logpi_d_exp = (
                    sum(
                        [
                            (pi * F.log_softmax(lg, dim=-1)).sum(dim=-1)
                            for pi, lg in zip(pi_list, d_logits)
                        ]
                    )
                    if d_logits is not None
                    else v1.new_zeros(v1.shape[0])
                )
                c_logp_s_agg = self._aggregate_continuous_logp(c_logp_s)
                if c_logp_s_agg is None:
                    c_logp_s_agg = v1.new_zeros(v1.shape[0])
                total_logp = c_logp_s_agg + logpi_d_exp
                current_logp_for_alpha = total_logp
                actor_loss = (alpha * total_logp - Vmin_pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            total_logp_detached = (
                current_logp_for_alpha.detach()
                if current_logp_for_alpha is not None
                else torch.zeros(obs_t.shape[0], device=self.device)
            )
            alpha_loss = -(
                self.log_alpha * (total_logp_detached + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

            if d_logits is not None and len(d_logits) > 0:
                pi_list = [F.softmax(lg, dim=-1) for lg in d_logits]
                terms = [
                    (pi * F.log_softmax(lg, dim=-1)).sum(dim=-1)
                    for pi, lg in zip(pi_list, d_logits)
                ]
                d_logpi_exp = terms[0]
                for t in terms[1:]:
                    d_logpi_exp = d_logpi_exp + t
                d_entropy = (-d_logpi_exp).mean()
            c_logp_agg = self._aggregate_continuous_logp(c_logp_s)
            if c_logp_agg is not None:
                c_entropy = (-c_logp_agg).mean()
                if c_logstd_logits is not None:
                    c_std = torch.exp(
                        torch.clamp(
                            c_logstd_logits,
                            min=self.log_std_clamp_range[0],
                            max=self.log_std_clamp_range[1],
                        )
                    )
                    c_std_mean = c_std.mean()
            if self.critic_mode == "V":
                if q_pi is not None:
                    if d_logits is not None:
                        d_actor_loss_val = alpha * (-d_entropy) - q_pi.detach().mean()
                    if c_logp_agg is not None:
                        c_actor_loss_val = (
                            alpha * c_logp_agg.mean() - q_pi.detach().mean()
                        )
            else:
                if Vmin_pi is not None:
                    if d_logits is not None:
                        d_actor_loss_val = (
                            alpha * (-d_entropy) - Vmin_pi.detach().mean()
                        )
                    if c_logp_agg is not None:
                        c_actor_loss_val = (
                            alpha * c_logp_agg.mean() - Vmin_pi.detach().mean()
                        )

        with torch.no_grad():
            cl = (critic_loss_1 + critic_loss_2) / 2
        rl_metrics = {
            "critic_loss": float(cl.item()),
            "d_actor_loss": float(d_actor_loss_val.detach().item()),
            "c_actor_loss": float(c_actor_loss_val.detach().item()),
            "d_entropy": float(d_entropy.detach().item()),
            "c_entropy": float(c_entropy.detach().item()),
            "c_std": float(c_std_mean.detach().item()),
        }
        return rl_metrics

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
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
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
        self.alpha_opt.load_state_dict(chkpt["alpha_opt"])
        if "log_alpha" in chkpt:
            with torch.no_grad():
                self.log_alpha.copy_(chkpt["log_alpha"].to(self.device).float())

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
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def _alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

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
