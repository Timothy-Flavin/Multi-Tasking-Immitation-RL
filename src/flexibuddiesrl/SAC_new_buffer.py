from .Agent import Agent, StochasticActor, ValueS, QS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
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
        hidden_dims=[256, 256],
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
        sac_tau=0.005,
        initial_temperature=0.2,
        mode="V",  # V or Q
    ):
        super(SAC, self).__init__()
        assert mode in ["Q", "V"], f"The critic mode needs to be 'V' or 'Q', you entered {mode}"
        if discrete_action_dims is None:
            mode = "V"
        self.critic_mode = mode
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
        self.discrete_action_dims = list(discrete_action_dims) if discrete_action_dims is not None else None
        
        self.has_continuous = (self.continuous_action_dim > 0)
        self.has_discrete = (self.discrete_action_dims is not None)
        
        self.action_dim = self.continuous_action_dim
        if self.has_discrete:
            self.action_dim += sum(self.discrete_action_dims)

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
        ).to(device)
        
        self.encoder = self.actor.encoder

        if self.critic_mode == "V":
            self._get_V_critics()
        else:
            self._get_Q_critics()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr * actor_ratio)
        self.Q1_opt = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.Q2_opt = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)

        _log_t0 = float(np.log(self.initial_temperature))
        self.log_alpha_c = torch.nn.Parameter(torch.tensor(_log_t0, device=self.device, dtype=torch.float32))
        self.log_alpha_d = torch.nn.Parameter(torch.tensor(_log_t0, device=self.device, dtype=torch.float32))
        self.alpha_opt_c = torch.optim.Adam([self.log_alpha_c], lr=self.lr)
        self.alpha_opt_d = torch.optim.Adam([self.log_alpha_d], lr=self.lr)

        self.target_entropy_c = -float(self.continuous_action_dim) if self.has_continuous else 0.0
        self.target_entropy_d = -float(np.sum(np.log(self.discrete_action_dims))) if self.has_discrete else 0.0
        
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

    def _get_Q_critics(self):
        common_kwargs = dict(
            obs_dim=self.obs_dim + self.continuous_action_dim,
            continuous_action_dim=0,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=copy.deepcopy(self.hidden_dims),
            encoder=None,
            activation=self.activation,
            orthogonal=self.orthogonal_init,
            dueling=True,
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

    def train_actions(self, observations, action_mask=None, step=False, debug=False) -> dict:
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device)
            
        is_3d = (len(observations.shape) == 3)
        if is_3d:
            n_agents, n_envs, obs_dim = observations.shape
            obs_flat = observations.reshape(-1, obs_dim)
        else:
            obs_flat = observations

        with torch.no_grad():
            continuous_means, continuous_log_std_logits, discrete_logits = self.actor(
                obs_flat, action_mask=action_mask, debug=debug
            )
            (
                discrete_actions,
                continuous_actions,
                _, _, _
            ) = self.actor.action_from_logits(
                continuous_means, continuous_log_std_logits, discrete_logits,
                gumbel=False, log_con=False, log_disc=False,
            )
            
            # Unflatten results if input was 3D
            if is_3d:
                if discrete_actions is not None:
                    discrete_actions = discrete_actions.reshape(n_agents, n_envs, -1)
                if continuous_actions is not None:
                    continuous_actions = continuous_actions.reshape(n_agents, n_envs, -1)

        return {
            "discrete_actions": discrete_actions.cpu().numpy() if discrete_actions is not None else None,
            "continuous_actions": continuous_actions.cpu().numpy() if continuous_actions is not None else None,
        }

    def reinforcement_learn(self, samples: ReplayBufferSamples, agent_num=0) -> dict:
        obs_t = samples.observations
        obs_next_t = samples.next_observations
        r_t = samples.rewards.squeeze(-1)
        d_t = samples.terminations.squeeze(-1)
        
        # Use the device of the data
        device = obs_t.device

        # Pull discrete and continuous actions from the buffer sample
        idx = 0
        discrete_actions = None
        if self.has_discrete:
            discrete_actions = obs_t.new_zeros((obs_t.shape[0], len(self.discrete_action_dims)), dtype=torch.long)
            for i in range(len(self.discrete_action_dims)):
                discrete_actions[:, i] = samples.actions[:, idx].long()
                idx += 1
        continuous_actions = samples.actions[:, idx:] if self.has_continuous else None

        alpha_c = self.log_alpha_c.exp()
        alpha_d = self.log_alpha_d.exp()
        is_v = (self.critic_mode == "V")

        # 1. Critic Update
        with torch.no_grad():
            c_means_n, c_logstd_logits_n, d_logits_n = self.actor(obs_next_t)
            d_act_n, c_act_n, d_logp_n, c_logp_n, _ = self.actor.action_from_logits(
                c_means_n, c_logstd_logits_n, d_logits_n,
                gumbel=True, log_con=self.has_continuous, log_disc=(is_v and self.has_discrete)
            )
            if is_v:
                a_next_vec = self._flatten_actions(c_act_n, d_act_n)
                q_next = torch.minimum(self.Q1_target(torch.cat([obs_next_t, a_next_vec], dim=-1)),
                                       self.Q2_target(torch.cat([obs_next_t, a_next_vec], dim=-1))).squeeze(-1)
                ent_pen_n = obs_next_t.new_zeros(obs_next_t.shape[0])
                if self.has_continuous: ent_pen_n += alpha_c * c_logp_n.sum(dim=-1)
                if self.has_discrete: ent_pen_n += alpha_d * self._discrete_neg_entropy(d_logits_n)
                v_next = q_next - ent_pen_n
            else:
                c_act_n_cat = torch.cat(c_act_n, dim=-1) if isinstance(c_act_n, list) else c_act_n
                in_vec_next = torch.cat([obs_next_t, c_act_n_cat], dim=-1) if self.has_continuous else obs_next_t
                v1_n, adv1_n, _ = self.Q1_target(in_vec_next)
                v2_n, adv2_n, _ = self.Q2_target(in_vec_next)
                v_next = torch.minimum(self._soft_v(v1_n, adv1_n, alpha_d), self._soft_v(v2_n, adv2_n, alpha_d))
                if self.has_continuous: v_next -= alpha_c * c_logp_n.sum(dim=-1)
            y = r_t + (1.0 - d_t) * (self.gamma * v_next)

        if is_v:
            a_vec = self._build_action_vector(continuous_actions, discrete_actions)
            q1 = self.Q1(torch.cat([obs_t, a_vec], dim=-1)).squeeze(-1)
            q2 = self.Q2(torch.cat([obs_t, a_vec], dim=-1)).squeeze(-1)
        else:
            in_vec = torch.cat([obs_t, continuous_actions], dim=-1) if self.has_continuous else obs_t
            v1, adv1, _ = self.Q1(in_vec)
            v2, adv2, _ = self.Q2(in_vec)
            q1 = v1.squeeze(-1) + self._gather_sum_adv(adv1, discrete_actions)
            q2 = v2.squeeze(-1) + self._gather_sum_adv(adv2, discrete_actions)

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.Q1_opt.zero_grad(); self.Q2_opt.zero_grad()
        critic_loss.backward()
        self.Q1_opt.step(); self.Q2_opt.step()

        self._soft_update(self.Q1_target, self.Q1, self.sac_tau)
        self._soft_update(self.Q2_target, self.Q2, self.sac_tau)

        # 2. Actor & Alpha Update
        res = {"critic_loss": critic_loss.item()}
        self._step_counter += 1
        if self._step_counter % self.actor_every == 0:
            c_means, c_logstd_logits, d_logits = self.actor(obs_t)
            d_act, c_act, d_logp, c_logp, _ = self.actor.action_from_logits(
                c_means, c_logstd_logits, d_logits,
                gumbel=True, log_con=self.has_continuous, log_disc=(is_v and self.has_discrete)
            )
            c_lp_agg = c_logp.sum(dim=-1) if self.has_continuous else None

            if is_v:
                a_vec_s = self._flatten_actions(c_act, d_act)
                q_pi = torch.minimum(self.Q1(torch.cat([obs_t, a_vec_s], dim=-1)),
                                     self.Q2(torch.cat([obs_t, a_vec_s], dim=-1))).squeeze(-1)
                ent_pen = obs_t.new_zeros(obs_t.shape[0])
                if self.has_continuous: ent_pen += alpha_c * c_lp_agg
                if self.has_discrete: ent_pen += alpha_d * self._discrete_neg_entropy(d_logits, detach=False)
                actor_loss = (ent_pen - q_pi).mean()
            else:
                c_act_s_cat = torch.cat(c_act, dim=-1) if isinstance(c_act, list) else c_act
                in_vec_s = torch.cat([obs_t, c_act_s_cat], dim=-1) if self.has_continuous else obs_t
                v1_s, adv1_s, _ = self.Q1(in_vec_s)
                v2_s, adv2_s, _ = self.Q2(in_vec_s)
                v_soft_min = torch.minimum(self._soft_v(v1_s, adv1_s, alpha_d), self._soft_v(v2_s, adv2_s, alpha_d))
                actor_loss = (alpha_c * (c_lp_agg if self.has_continuous else 0) - v_soft_min).mean()
                if self.has_discrete:
                    for i, logit_i in enumerate(d_logits):
                        pi_i = F.softmax(logit_i, dim=-1)
                        log_pi_i = F.log_softmax(logit_i, dim=-1)
                        q_min_i = torch.minimum(v1_s + adv1_s[i], v2_s + adv2_s[i])
                        actor_loss += (pi_i * (log_pi_i - F.log_softmax(q_min_i / alpha_d.detach(), dim=-1).detach())).sum(dim=-1).mean()

            self.actor_opt.zero_grad(); actor_loss.backward()
            self.actor_opt.step()

            # Alpha Update
            if self.has_continuous:
                alpha_c_loss = -(self.log_alpha_c * (c_lp_agg.detach().mean() + self.target_entropy_c))
                self.alpha_opt_c.zero_grad(); alpha_c_loss.backward(); self.alpha_opt_c.step()
            if self.has_discrete:
                d_neg_ent = self._discrete_neg_entropy(d_logits).mean()
                alpha_d_loss = -(self.log_alpha_d * (d_neg_ent + self.target_entropy_d))
                self.alpha_opt_d.zero_grad(); alpha_d_loss.backward(); self.alpha_opt_d.step()
            
            res.update({"actor_loss": actor_loss.item(), "alpha_c": alpha_c.item(), "alpha_d": alpha_d.item()})

        return res

    def imitation_learn(self, observations, continuous_actions, discrete_actions, action_mask=None, debug=False):
        return {"rl_actor_loss": 0, "rl_critic_loss": 0}

    def param_count(self) -> tuple[int, int]:
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.Q1.parameters()) * 2
        return actor_params, actor_params + critic_params

    def ego_actions(self, observations, action_mask=None):
        return self.train_actions(observations, action_mask=action_mask)

    def expected_V(self, obs, legal_action=None):
        if not torch.is_tensor(obs):
            obs = T(obs, device=self.device)
        with torch.no_grad():
            c_means, c_logstd, d_logits = self.actor(obs)
            d_act, c_act, d_logp, c_logp, _ = self.actor.action_from_logits(
                c_means, c_logstd, d_logits, gumbel=False, log_con=False, log_disc=False
            )
            a_vec = self._flatten_actions(c_act, d_act)
            v = torch.minimum(self.Q1(torch.cat([obs, a_vec], dim=-1)),
                              self.Q2(torch.cat([obs, a_vec], dim=-1))).squeeze(-1)
            return v

    def stable_greedy(self, obs, legal_action):
        acts = self.train_actions(obs, action_mask=legal_action)
        adiscrete = torch.as_tensor(acts["discrete_actions"], device=self.device) if acts["discrete_actions"] is not None else None
        acontinuous = torch.as_tensor(acts["continuous_actions"], device=self.device) if acts["continuous_actions"] is not None else None
        return adiscrete, acontinuous

    def utility_function(self, observations, actions=None):
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device)
        if actions is None:
            return None # Requires actions for Q-values
        a_vec = self._build_action_vector(actions[0], actions[1]) # Assuming [c, d]
        q = torch.minimum(self.Q1(torch.cat([observations, a_vec], dim=-1)),
                          self.Q2(torch.cat([observations, a_vec], dim=-1))).squeeze(-1)
        return q

    def _flatten_actions(self, c_act, d_act):
        parts = []
        if c_act is not None: parts.append(c_act if c_act.ndim > 1 else c_act.unsqueeze(0))
        if d_act is not None:
            if isinstance(d_act, list): parts.extend(d_act)
            else: parts.append(d_act)
        return torch.cat(parts, dim=-1)

    def _build_action_vector(self, c_actions, d_actions):
        parts = []
        if self.has_continuous: parts.append(c_actions)
        if self.has_discrete:
            for i, dim in enumerate(self.discrete_action_dims):
                parts.append(F.one_hot(d_actions[:, i], dim).float())
        return torch.cat(parts, dim=-1)

    def _soft_v(self, v, adv_heads, alpha):
        sv = v.squeeze(-1)
        if adv_heads is not None:
            for adv in adv_heads:
                sv = sv + alpha * torch.logsumexp(adv / alpha, dim=-1)
        return sv

    def _discrete_neg_entropy(self, d_logits, detach=True):
        ent = 0
        for lg in d_logits:
            logits = lg.detach() if detach else lg
            pi = F.softmax(logits, dim=-1)
            log_pi = F.log_softmax(logits, dim=-1)
            ent += (pi * log_pi).sum(dim=-1)
        return ent

    @staticmethod
    def _gather_sum_adv(adv_heads, d_idx):
        res = 0
        for i, adv in enumerate(adv_heads):
            res += adv.gather(1, d_idx[:, i:i+1].long()).squeeze(-1)
        return res

    def _hard_update(self, target, source): target.load_state_dict(source.state_dict())
    def _soft_update(self, target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.lerp_(s.data, tau)

    def save(self, path): torch.save(self.state_dict(), path)
    def load(self, path): self.load_state_dict(torch.load(path, map_location=self.device))
