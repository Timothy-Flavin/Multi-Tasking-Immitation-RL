from .Agent import Agent, StochasticActor, ValueS, QS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Union
import copy
from .buffers import ReplayBuffer, ReplayBufferSamples
from .Util import T


class SAC(nn.Module, Agent):
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
        target_discrete_entropy_percentage=0.5,
    ):
        super(SAC, self).__init__()
        assert mode in [
            "Q",
            "V",
        ], f"The critic mode needs to be 'V' or 'Q', you entered {mode}"
        if discrete_action_dims is None:
            mode = "V"
        self.critic_mode = mode
        self.target_discrete_entropy_percentage = target_discrete_entropy_percentage
        self.log_std_clamp_range = log_std_clamp_range
        self.actor_every = actor_every
        self.actor_ratio = actor_ratio
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.orthogonal_init = orthogonal_init
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

        self.has_continuous = self.continuous_action_dim > 0
        self.has_discrete = self.discrete_action_dims is not None
        self.has_actor = self.has_continuous or (
            self.critic_mode == "V" and self.has_discrete
        )

        self.action_dim = self.continuous_action_dim
        if self.has_discrete:
            self.action_dim += sum(self.discrete_action_dims)

        # In Q mode, the actor should NEVER output discrete actions.
        # The critic handles discrete evaluation directly.
        self.actor_discrete_dims = (
            self.discrete_action_dims if self.critic_mode == "V" else None
        )
        self.actor = None
        if self.has_actor:
            self.actor = StochasticActor(
                obs_dim=obs_dim,
                continuous_action_dim=continuous_action_dim,
                discrete_action_dims=self.actor_discrete_dims,
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
            ).to(device)
        self.encoder = encoder

        if self.critic_mode == "V":
            self._get_V_critics()
        else:
            self._get_Q_critics()

        if self.has_actor:
            self.actor_opt = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr * actor_ratio
            )
        self.Q1_opt = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.Q2_opt = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)

        _log_t0 = float(np.log(self.initial_temperature))
        self.log_alpha_c = torch.nn.Parameter(
            torch.tensor(_log_t0, device=self.device, dtype=torch.float32)
        )
        self.log_alpha_d = torch.nn.Parameter(
            torch.tensor(_log_t0, device=self.device, dtype=torch.float32)
        )
        self.alpha_opt_c = torch.optim.Adam([self.log_alpha_c], lr=self.lr)
        self.alpha_opt_d = torch.optim.Adam([self.log_alpha_d], lr=self.lr)

        self.target_entropy_c = (
            -float(self.continuous_action_dim) if self.has_continuous else 0.0
        )
        self.target_entropy_d = 0.0
        if self.has_discrete and self.discrete_action_dims is not None:
            self.target_entropy_d = (
                self.target_discrete_entropy_percentage
                * float(np.sum(np.log(self.discrete_action_dims)))
                if self.has_discrete
                else 0.0
            )

        self._step_counter = 0

    def _get_V_critics(self):
        common_kwargs = dict(
            obs_dim=self.obs_dim + self.action_dim,
            hidden_dim=self.hidden_dims[0],
            device=self.device,
            activation=self.activation,
            orthogonal_init=self.orthogonal_init,
        )
        self.Q1 = ValueS(**common_kwargs).to(self.device)
        self.Q2 = ValueS(**common_kwargs).to(self.device)
        self.Q1_target = ValueS(**common_kwargs).to(self.device)
        self.Q2_target = ValueS(**common_kwargs).to(self.device)
        self._hard_update(self.Q1_target, self.Q1)
        self._hard_update(self.Q2_target, self.Q2)
        for p in self.Q1_target.parameters():
            p.requires_grad_(False)
        for p in self.Q2_target.parameters():
            p.requires_grad_(False)
        self.Q1_target.eval()
        self.Q2_target.eval()

    def _get_Q_critics(self):
        common_kwargs = dict(
            obs_dim=self.obs_dim + self.continuous_action_dim,
            continuous_action_dim=0,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=self.hidden_dims,
            encoder=None,
            activation=self.activation,
            orthogonal=self.orthogonal_init,
            dueling=True,  # Enforce direct Q heads, no dueling
            device=self.device,
            n_c_action_bins=0,
            QMIX=False,
        )
        self.Q1 = QS(**common_kwargs).to(self.device)
        self.Q2 = QS(**common_kwargs).to(self.device)
        self.Q1_target = QS(**common_kwargs).to(self.device)
        self.Q2_target = QS(**common_kwargs).to(self.device)
        self._hard_update(self.Q1_target, self.Q1)
        self._hard_update(self.Q2_target, self.Q2)
        for p in self.Q1_target.parameters():
            p.requires_grad_(False)
        for p in self.Q2_target.parameters():
            p.requires_grad_(False)
        self.Q1_target.eval()
        self.Q2_target.eval()

    def train_actions(
        self, observations, action_mask=None, step=False, debug=False
    ) -> dict:
        was_tensor = torch.is_tensor(observations)
        if not was_tensor:
            observations = T(observations, device=self.device)

        is_3d = len(observations.shape) == 3
        if is_3d:
            n_envs, n_agents, obs_dim = observations.shape
            obs_flat = observations.reshape(-1, obs_dim)
        else:
            obs_flat = observations

        with torch.no_grad():
            if self.has_actor:
                c_means, c_logstd_logits, d_logits = self.actor(
                    obs_flat, action_mask=action_mask, debug=debug
                )

            if not self.has_actor:
                # Q mode, discrete-only: sample directly from critic
                _, d_q_heads, _ = self.Q1(obs_flat)
                d_acts = []
                for q_head in d_q_heads:
                    probs = F.softmax(q_head / self.log_alpha_d.exp().detach(), dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    d_acts.append(dist.sample().unsqueeze(-1))
                discrete_actions = torch.cat(d_acts, dim=-1)
                continuous_actions = None
            elif self.critic_mode == "Q" and self.has_discrete:
                _, continuous_actions, _, _, _ = self.actor.action_from_logits(
                    c_means, c_logstd_logits, None, gumbel=False, log_con=False
                )
                q_in = (
                    torch.cat([obs_flat, continuous_actions], dim=-1)
                    if self.has_continuous
                    else obs_flat
                )
                _, d_q_heads, _ = self.Q1(q_in)
                d_acts = []
                for q_head in d_q_heads:
                    probs = F.softmax(q_head / self.log_alpha_d.exp().detach(), dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    d_acts.append(dist.sample().unsqueeze(-1))
                discrete_actions = torch.cat(d_acts, dim=-1)
            else:
                discrete_actions, continuous_actions, _, _, _ = (
                    self.actor.action_from_logits(
                        c_means, c_logstd_logits, d_logits, gumbel=False, log_con=False
                    )
                )

            if is_3d:
                if discrete_actions is not None:
                    discrete_actions = discrete_actions.reshape(n_envs, n_agents, -1)
                if continuous_actions is not None:
                    continuous_actions = continuous_actions.reshape(
                        n_envs, n_agents, -1
                    )

        def _maybe_numpy(x):
            if was_tensor or x is None:
                return x
            return x.cpu().numpy()

        return {
            "discrete_actions": _maybe_numpy(discrete_actions),
            "continuous_actions": _maybe_numpy(continuous_actions),
        }

    def reinforcement_learn(
        self, samples: ReplayBufferSamples, agent_num=0, critic_only=False
    ) -> dict:
        obs_t = samples.observations
        obs_next_t = samples.next_observations
        r_t = samples.rewards.squeeze(-1)
        d_t = samples.terminations.squeeze(-1)

        discrete_actions = None
        idx = 0
        if self.has_discrete:
            idx = len(self.discrete_action_dims)
            discrete_actions = samples.actions[:, :idx].long()
        continuous_actions = samples.actions[:, idx:] if self.has_continuous else None

        alpha_c = self.log_alpha_c.exp()
        alpha_d = self.log_alpha_d.exp()
        is_v = self.critic_mode == "V"

        # ------------------------------------
        # 1. Critic Update
        # ------------------------------------
        with torch.no_grad():
            if self.has_actor:
                c_means_n, c_logstd_logits_n, d_logits_n = self.actor(obs_next_t)
                d_act_n, c_act_n, _, c_logp_n, _ = self.actor.action_from_logits(
                    c_means_n,
                    c_logstd_logits_n,
                    d_logits_n,
                    gumbel=is_v,
                    log_con=self.has_continuous,
                    log_disc=(is_v and self.has_discrete),
                )
            else:
                c_act_n, c_logp_n = None, None

            if is_v:
                a_next_vec = self._flatten_actions(c_act_n, d_act_n)
                q_in_next = torch.cat([obs_next_t, a_next_vec], dim=-1)
                q_next = torch.minimum(
                    self.Q1_target(q_in_next), self.Q2_target(q_in_next)
                ).squeeze(-1)
                ent_pen_n = obs_next_t.new_zeros(obs_next_t.shape[0])
                if self.has_continuous:
                    ent_pen_n += alpha_c * self._aggregate_continuous_logp(c_logp_n)
                if self.has_discrete:
                    ent_pen_n += alpha_d * self._discrete_neg_entropy(d_logits_n)
                v_next = q_next - ent_pen_n
            else:
                in_vec_next = (
                    torch.cat([obs_next_t, c_act_n], dim=-1)
                    if self.has_continuous
                    else obs_next_t
                )
                v1_next, q1_next_heads, _ = self.Q1_target(in_vec_next)
                v2_next, q2_next_heads, _ = self.Q2_target(in_vec_next)

                # Value heads: zeros when dueling=False, V(s') when dueling=True
                v_next = torch.minimum(v1_next, v2_next).squeeze(-1)
                if self.has_discrete:
                    for q1_h, q2_h in zip(q1_next_heads, q2_next_heads):
                        q_min_h = torch.minimum(q1_h, q2_h)
                        # Soft V = E_pi[A] + alpha*H  (direct form; stable as alpha->0)
                        logprobs = F.log_softmax(
                            q_min_h / alpha_d.clamp(min=1e-8), dim=-1
                        )
                        probs = logprobs.exp()
                        v_next += (probs * q_min_h).sum(dim=-1) - alpha_d * (
                            probs * logprobs
                        ).sum(dim=-1)

                if self.has_continuous:
                    v_next -= alpha_c * self._aggregate_continuous_logp(c_logp_n)

            y = r_t + (1.0 - d_t) * (self.gamma * v_next)

        if is_v:
            a_vec = self._build_action_vector(continuous_actions, discrete_actions)
            q_in = torch.cat([obs_t, a_vec], dim=-1)
            q1 = self.Q1(q_in).squeeze(-1)
            q2 = self.Q2(q_in).squeeze(-1)
        else:
            in_vec = (
                torch.cat([obs_t, continuous_actions], dim=-1)
                if self.has_continuous
                else obs_t
            )
            v1, q1_heads, _ = self.Q1(in_vec)
            v2, q2_heads, _ = self.Q2(in_vec)

            # Seed with value heads (zeros when dueling=False, V(s) when dueling=True)
            q1 = v1.squeeze(-1)
            q2 = v2.squeeze(-1)
            if self.has_discrete:
                for i, (q1_h, q2_h) in enumerate(zip(q1_heads, q2_heads)):
                    a_d_i = discrete_actions[:, i : i + 1].long()
                    q1 = q1 + q1_h.gather(1, a_d_i).squeeze(-1)
                    q2 = q2 + q2_h.gather(1, a_d_i).squeeze(-1)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.Q1_opt.zero_grad(set_to_none=True)
        self.Q2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0)
        self.Q1_opt.step()
        self.Q2_opt.step()

        self._soft_update(self.Q1_target, self.Q1, self.sac_tau)
        self._soft_update(self.Q2_target, self.Q2, self.sac_tau)

        # ------------------------------------
        # 2. Actor & Alpha Update
        # ------------------------------------
        res = {"critic_loss": critic_loss.item()}
        self._step_counter += 1
        if not critic_only and self._step_counter % self.actor_every == 0:
            d_neg_ent_q_mode = None

            if self.has_actor:
                c_means, c_logstd_logits, d_logits = self.actor(obs_t)
                d_act, c_act, _, c_logp, _ = self.actor.action_from_logits(
                    c_means,
                    c_logstd_logits,
                    d_logits,
                    gumbel=is_v,
                    log_con=self.has_continuous,
                    log_disc=(is_v and self.has_discrete),
                )
                c_lp_agg = (
                    self._aggregate_continuous_logp(c_logp)
                    if self.has_continuous
                    else None
                )

                d_neg_ent_q_mode = 0.0

                if is_v:
                    a_vec_s = self._flatten_actions(c_act, d_act)
                    q_in_s = torch.cat([obs_t, a_vec_s], dim=-1)
                    q_pi = torch.minimum(self.Q1(q_in_s), self.Q2(q_in_s)).squeeze(-1)
                    ent_pen = obs_t.new_zeros(obs_t.shape[0])
                    if self.has_continuous:
                        ent_pen += alpha_c * c_lp_agg
                    if self.has_discrete:
                        ent_pen += alpha_d * self._discrete_neg_entropy(
                            d_logits, detach=False
                        )
                    actor_loss = (ent_pen - q_pi).mean()
                else:
                    in_vec_s = (
                        torch.cat([obs_t, c_act], dim=-1)
                        if self.has_continuous
                        else obs_t
                    )
                    v1_s, q1_heads_s, _ = self.Q1(in_vec_s)
                    v2_s, q2_heads_s, _ = self.Q2(in_vec_s)

                    # Start from V(s, a_c); add E_pi_d[A] per head
                    q_s_tot = torch.minimum(v1_s, v2_s).squeeze(-1)
                    if self.has_discrete:
                        for q1_h, q2_h in zip(q1_heads_s, q2_heads_s):
                            q_min_h = torch.minimum(q1_h, q2_h)
                            probs = F.softmax(
                                q_min_h.detach() / alpha_d.detach(), dim=-1
                            )
                            log_probs = torch.log(probs + 1e-8)
                            d_neg_ent_q_mode += (probs * log_probs).sum(dim=-1)
                            q_s_tot += (probs * q_min_h).sum(dim=-1)

                    actor_loss = (
                        alpha_c * (c_lp_agg if self.has_continuous else 0) - q_s_tot
                    ).mean()

                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_opt.step()

                if self.has_continuous:
                    alpha_c_loss = -(
                        self.log_alpha_c
                        * (c_lp_agg.detach().mean() + self.target_entropy_c)
                    )
                    self.alpha_opt_c.zero_grad(set_to_none=True)
                    alpha_c_loss.backward()
                    self.alpha_opt_c.step()

            # Alpha_d update: runs even without actor (discrete-only Q mode)
            if self.has_discrete:
                if is_v:
                    d_neg_ent = self._discrete_neg_entropy(d_logits).mean()
                elif d_neg_ent_q_mode is not None:
                    d_neg_ent = d_neg_ent_q_mode.mean()
                else:
                    # No actor, Q mode: derive entropy estimate from critic directly
                    _, q_heads_s, _ = self.Q1(obs_t)
                    d_neg_ent_val = obs_t.new_zeros(obs_t.shape[0])
                    for q_h in q_heads_s:
                        probs = F.softmax(q_h.detach() / alpha_d.detach(), dim=-1)
                        log_probs = torch.log(probs + 1e-8)
                        d_neg_ent_val += (probs * log_probs).sum(dim=-1)
                    d_neg_ent = d_neg_ent_val.mean()
                alpha_d_loss = -(self.log_alpha_d * (d_neg_ent + self.target_entropy_d))
                self.alpha_opt_d.zero_grad(set_to_none=True)
                alpha_d_loss.backward()
                self.alpha_opt_d.step()

            # Gumbel temperature decay is strictly a V mode mechanism
            if self.has_actor and self.has_discrete and is_v:
                self.actor.gumbel_tau = max(
                    self.actor.gumbel_tau_min,
                    self.actor.gumbel_tau * self.actor.gumbel_tau_decay,
                )

            if self.has_actor:
                res.update(
                    {
                        "actor_loss": actor_loss.item(),
                        "alpha_c": alpha_c.item(),
                        "alpha_d": alpha_d.item(),
                        "gumbel_tau": self.actor.gumbel_tau if is_v else None,
                    }
                )
            else:
                res.update({"alpha_d": alpha_d.item()})

        return res

    def imitation_learn(
        self,
        observations,
        continuous_actions,
        discrete_actions,
        action_mask=None,
        debug=False,
    ):
        return {"rl_actor_loss": 0, "rl_critic_loss": 0}

    def param_count(self) -> tuple[int, int]:
        actor_params = (
            sum(p.numel() for p in self.actor.parameters()) if self.has_actor else 0
        )
        critic_params = sum(p.numel() for p in self.Q1.parameters()) * 2
        return actor_params, actor_params + critic_params

    def tonumpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to("cpu").numpy()
        return x

    def ego_actions(self, observations, action_mask=None) -> dict:
        with torch.no_grad():
            obs_t = T(observations, self.device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)

            if self.has_actor:
                c_means, _, d_logits = self.actor(
                    obs_t, action_mask=action_mask, debug=False
                )
                if c_means is not None:
                    c_act = torch.tanh(c_means)
                    if self.max_actions is not None and self.min_actions is not None:
                        center = (self.max_actions + self.min_actions) / 2.0
                        half_range = (self.max_actions - self.min_actions) / 2.0
                        c_act = center + half_range * c_act
                else:
                    c_act = None
            else:
                c_act = None
                d_logits = None

            # Discrete branching logic
            if self.critic_mode == "Q" and self.has_discrete:
                in_vec = (
                    torch.cat([obs_t, c_act], dim=-1) if self.has_continuous else obs_t
                )
                _, d_q_heads, _ = self.Q1(in_vec)
                d_actions_idx = [torch.argmax(q_head, dim=-1) for q_head in d_q_heads]
            else:
                if d_logits is not None:
                    d_actions_idx = [torch.argmax(logit, dim=-1) for logit in d_logits]
                else:
                    d_actions_idx = None

        return {"discrete_action": d_actions_idx, "continuous_action": c_act}

    def expected_V(self, obs, legal_action=None):
        if not torch.is_tensor(obs):
            obs = T(obs, device=self.device)
        with torch.no_grad():
            alpha_c = self.log_alpha_c.exp()
            alpha_d = self.log_alpha_d.exp()
            if self.has_actor:
                c_means, c_logstd, d_logits = self.actor(obs)
                d_act, c_act, _, c_logp, _ = self.actor.action_from_logits(
                    c_means,
                    c_logstd,
                    d_logits,
                    gumbel=True,
                    log_con=self.has_continuous,
                    log_disc=(self.critic_mode == "V" and self.has_discrete),
                )
            else:
                c_act, c_logp = None, None
            if self.critic_mode == "V":
                a_vec = self._flatten_actions(c_act, d_act)
                q_in = torch.cat([obs, a_vec], dim=-1)
                v = torch.minimum(self.Q1_target(q_in), self.Q2_target(q_in)).squeeze(
                    -1
                )
                if self.has_continuous and c_logp is not None:
                    v -= alpha_c * self._aggregate_continuous_logp(c_logp)
                if self.has_discrete and d_logits is not None:
                    v -= alpha_d * self._discrete_neg_entropy(d_logits)
            else:
                in_vec = torch.cat([obs, c_act], dim=-1) if self.has_continuous else obs
                _, q1_heads, _ = self.Q1_target(in_vec)
                _, q2_heads, _ = self.Q2_target(in_vec)

                v = obs.new_zeros(obs.shape[0])
                if self.has_discrete:
                    for q1_h, q2_h in zip(q1_heads, q2_heads):
                        q_min_h = torch.minimum(q1_h, q2_h)
                        v += q_min_h.max(dim=-1)[0]
                if self.has_continuous and c_logp is not None:
                    v -= alpha_c * self._aggregate_continuous_logp(c_logp)
            return v

    def stable_greedy(self, obs, legal_action):
        acts = self.train_actions(obs, action_mask=legal_action)
        adiscrete = (
            torch.as_tensor(acts["discrete_actions"], device=self.device)
            if acts["discrete_actions"] is not None
            else None
        )
        acontinuous = (
            torch.as_tensor(acts["continuous_actions"], device=self.device)
            if acts["continuous_actions"] is not None
            else None
        )
        return adiscrete, acontinuous

    def utility_function(self, observations, actions=None):
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device)
        if actions is None:
            return None

        if self.critic_mode == "V":
            a_vec = self._build_action_vector(actions[1], actions[0])
            q_in = torch.cat([observations, a_vec], dim=-1)
            q = torch.minimum(self.Q1(q_in), self.Q2(q_in)).squeeze(-1)
            return q
        else:
            in_vec = (
                torch.cat([observations, actions[1]], dim=-1)
                if self.has_continuous
                else observations
            )
            _, q1_heads, _ = self.Q1(in_vec)
            _, q2_heads, _ = self.Q2(in_vec)

            q_tot = observations.new_zeros(observations.shape[0])
            if self.has_discrete:
                for i, (q1_h, q2_h) in enumerate(zip(q1_heads, q2_heads)):
                    a_d_i = actions[0][:, i : i + 1].long()
                    q_min_h = torch.minimum(q1_h, q2_h)
                    q_tot += q_min_h.gather(1, a_d_i).squeeze(-1)
            return q_tot

    def _flatten_actions(self, c_act, d_act):
        parts = []
        if c_act is not None:
            parts.append(c_act if c_act.ndim > 1 else c_act.unsqueeze(0))
        if d_act is not None:
            if isinstance(d_act, list):
                parts.extend(d_act)
            else:
                parts.append(d_act)
        return torch.cat(parts, dim=-1)

    def _build_action_vector(self, c_actions, d_actions):
        parts = []
        if self.has_continuous:
            parts.append(c_actions)
        if self.has_discrete:
            for i, dim in enumerate(self.discrete_action_dims):
                parts.append(F.one_hot(d_actions[:, i], dim).float())
        return torch.cat(parts, dim=-1)

    def _aggregate_continuous_logp(self, c_logp):
        if c_logp is None:
            return None
        if isinstance(c_logp, (list, tuple)):
            xs = [t if t.ndim == 1 else t.sum(dim=-1) for t in c_logp]
            out = xs[0]
            for t in xs[1:]:
                out = out + t
            return out
        return c_logp.sum(dim=-1) if c_logp.ndim > 1 else c_logp

    def _discrete_neg_entropy(self, d_logits, detach=True):
        ent = 0
        for lg in d_logits:
            logits = lg.detach() if detach else lg
            log_pi = F.log_softmax(logits, dim=-1)
            ent += (log_pi.exp() * log_pi).sum(dim=-1)
        return ent

    def _hard_update(self, target, source):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.lerp_(s.data, tau)

    def save(self, checkpoint_path):
        state = {
            "Q1": self.Q1.state_dict(),
            "Q1_target": self.Q1_target.state_dict(),
            "Q2": self.Q2.state_dict(),
            "Q2_target": self.Q2_target.state_dict(),
            "Q1_opt": self.Q1_opt.state_dict(),
            "Q2_opt": self.Q2_opt.state_dict(),
            "alpha_opt_c": self.alpha_opt_c.state_dict(),
            "alpha_opt_d": self.alpha_opt_d.state_dict(),
            "log_alpha_c": self.log_alpha_c.detach().cpu(),
            "log_alpha_d": self.log_alpha_d.detach().cpu(),
        }
        if self.has_actor:
            state["actor"] = self.actor.state_dict()
            state["actor_opt"] = self.actor_opt.state_dict()
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        chkpt = torch.load(checkpoint_path, map_location=self.device)
        if self.has_actor and "actor" in chkpt:
            self.actor.load_state_dict(chkpt["actor"])
            self.actor_opt.load_state_dict(chkpt["actor_opt"])
        self.Q1.load_state_dict(chkpt["Q1"])
        self.Q1_target.load_state_dict(chkpt["Q1_target"])
        self.Q2.load_state_dict(chkpt["Q2"])
        self.Q2_target.load_state_dict(chkpt["Q2_target"])
        self.Q1_opt.load_state_dict(chkpt["Q1_opt"])
        self.Q2_opt.load_state_dict(chkpt["Q2_opt"])
        if "alpha_opt_c" in chkpt:
            self.alpha_opt_c.load_state_dict(chkpt["alpha_opt_c"])
            self.alpha_opt_d.load_state_dict(chkpt["alpha_opt_d"])
        if "log_alpha_c" in chkpt:
            with torch.no_grad():
                self.log_alpha_c.copy_(chkpt["log_alpha_c"].to(self.device).float())
                self.log_alpha_d.copy_(chkpt["log_alpha_d"].to(self.device).float())
