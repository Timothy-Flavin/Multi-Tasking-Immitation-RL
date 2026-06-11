import numpy as np
import random
import torch.nn as nn
import torch
from torch.distributions import Categorical
from .Agent import Agent, QS
from .buffers import ReplayBuffer, ReplayBufferSamples
import os
import pickle
import torch.nn.functional as F
from .Util import T
import copy
from enum import Enum


class dqntype(Enum):
    EGreedy = 0
    Soft = 1
    Munchausen = 2


class DQN(nn.Module, Agent):
    def __init__(
        self,
        obs_dim=10,
        continuous_action_dims=0,
        discrete_action_dims=[2],
        max_actions=None,
        min_actions=None,
        lr=1e-3,
        gamma=0.99,
        device="cpu",
        hidden_dims=[256, 256],
        activation="relu",
        n_c_action_bins=11,
        tau=0.005,
        dueling=True,
        target_update_interval=1,
        batch_size=256,
        load_from_checkpoint=None,
        name="DQN",
        eval_mode=False,
        head_hidden_dims=[128],
        encoder=None,
        mix_type=None,
        QMIX_hidden_dim=64,
        orthogonal=True,
        entropy=0.0,
        munchausen=0.0,
        imitation_type="cross_entropy",
        imitation_lr=1e-5,
    ):
        super(DQN, self).__init__()
        self.config = locals()
        self.config.pop("self")
        
        self.obs_dim = obs_dim
        self.continuous_action_dims = continuous_action_dims
        self.discrete_action_dims = discrete_action_dims
        self.max_actions = max_actions
        self.min_actions = min_actions
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.tau = tau
        self.dueling = dueling
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.n_c_action_bins = n_c_action_bins
        self.name = name
        self.eval_mode = eval_mode
        self.mix_type = mix_type.upper() if mix_type is not None else None
        self.QMIX = (self.mix_type == "QMIX")
        self.QMIX_hidden_dim = QMIX_hidden_dim
        
        self.np_max_actions = np.array(max_actions) if max_actions is not None else None
        self.np_min_actions = np.array(min_actions) if min_actions is not None else None

        # Soft / Munchausen DQN setup
        self.entropy_loss_coef = entropy
        self.munchausen = munchausen
        self.imitation_type = imitation_type
        self.dqn_type = dqntype.EGreedy
        if entropy > 0:
            self.dqn_type = dqntype.Soft
        if entropy > 0 and munchausen > 0:
            self.dqn_type = dqntype.Munchausen

        # Mean-centred action scale helpers (needed for soft/Munchausen conversion)
        if self.np_max_actions is not None and self.np_min_actions is not None:
            self.np_action_ranges = self.np_max_actions - self.np_min_actions
            self.np_action_means = (self.np_max_actions + self.np_min_actions) / 2.0
            # Cached device tensors — avoids re-allocating on every train_actions call
            self._max_t       = torch.tensor(self.np_max_actions, device=device, dtype=torch.float32)
            self._min_t       = torch.tensor(self.np_min_actions, device=device, dtype=torch.float32)
            self._bin_widths  = (self._max_t - self._min_t) / (self.n_c_action_bins - 1)
        else:
            self.np_action_ranges = None
            self.np_action_means  = None
            self._max_t = self._min_t = self._bin_widths = None

        self._get_torch_params(encoder, hidden_dims, head_hidden_dims, activation, orthogonal)
        # Separate imitation optimizer so IL steps don't corrupt RL momentum
        self.imitation_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=imitation_lr)

        if load_from_checkpoint:
            self.load(load_from_checkpoint)

    def _get_torch_params(self, encoder, hidden_dims, head_hidden_dims, activation, orthogonal):
        self.q_net = QS(
            obs_dim=self.obs_dim,
            continuous_action_dim=self.continuous_action_dims,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=copy.deepcopy(hidden_dims),
            activation=activation,
            dueling=self.dueling,
            device=self.device,
            n_c_action_bins=self.n_c_action_bins,
            head_hidden_dims=copy.deepcopy(head_hidden_dims),
            encoder=encoder,
            QMIX=self.QMIX,
            QMIX_hidden_dim=self.QMIX_hidden_dim,
            orthogonal=orthogonal
        ).to(self.device)
        
        self.target_q_net = QS(
            obs_dim=self.obs_dim,
            continuous_action_dim=self.continuous_action_dims,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dims=copy.deepcopy(hidden_dims),
            activation=activation,
            dueling=self.dueling,
            device=self.device,
            n_c_action_bins=self.n_c_action_bins,
            head_hidden_dims=copy.deepcopy(head_hidden_dims),
            encoder=encoder,
            QMIX=self.QMIX,
            QMIX_hidden_dim=self.QMIX_hidden_dim,
            orthogonal=orthogonal
        ).to(self.device)
        
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.steps = 0

    def train_actions(self, observations, action_mask=None, step=False, epsilon=0.1):
        if step: self.steps += 1

        was_tensor = torch.is_tensor(observations)
        if not was_tensor:
            observations = T(observations, device=self.device)

        is_3d = (len(observations.shape) == 3)
        if is_3d:
            n_envs, n_agents, obs_dim = observations.shape
            obs_flat = observations.reshape(-1, obs_dim)
        else:
            obs_flat = observations

        with torch.no_grad():
            values_flat, d_adv, c_adv = self.q_net(obs_flat)

            d_actions = []
            if d_adv is not None:
                for head in d_adv:
                    if random.random() < epsilon and not self.eval_mode:
                        d_actions.append(torch.randint(0, head.shape[-1], (obs_flat.shape[0],), device=self.device))
                    elif self.dqn_type in (dqntype.Soft, dqntype.Munchausen):
                        d_actions.append(Categorical(logits=head / self.entropy_loss_coef).sample())
                    else:
                        d_actions.append(head.argmax(dim=-1))
                d_actions = torch.stack(d_actions, dim=-1)
            else:
                d_actions = None

            c_actions_cont = None
            if c_adv is not None:
                c_actions_bins = []
                for head in c_adv:
                    if random.random() < epsilon and not self.eval_mode:
                        c_actions_bins.append(torch.randint(0, head.shape[-1], (obs_flat.shape[0],), device=self.device))
                    elif self.dqn_type in (dqntype.Soft, dqntype.Munchausen):
                        c_actions_bins.append(Categorical(logits=head / self.entropy_loss_coef).sample())
                    else:
                        c_actions_bins.append(head.argmax(dim=-1))
                c_actions_bins = torch.stack(c_actions_bins, dim=-1) # [B, n_c_dims]

                # Convert bin indices to continuous values using cached tensors
                c_actions_cont = self._min_t + c_actions_bins.float() * self._bin_widths

            # Unflatten results if input was 3D
            if is_3d:
                if d_actions is not None:
                    d_actions = d_actions.reshape(n_envs, n_agents, -1)
                if c_actions_cont is not None:
                    c_actions_cont = c_actions_cont.reshape(n_envs, n_agents, -1)
                values = values_flat.reshape(n_envs, n_agents, -1)
            else:
                values = values_flat

        def _maybe_numpy(x):
            if was_tensor or x is None:
                return x
            return x.cpu().numpy()

        return {
            "discrete_actions": _maybe_numpy(d_actions),
            "continuous_actions": _maybe_numpy(c_actions_cont),
            "values": _maybe_numpy(values)
        }

    def reinforcement_learn(self, samples: ReplayBufferSamples, agent_num=0):
        if self.eval_mode: return {"rl_loss": 0}
        
        obs = samples.observations
        actions = samples.actions # [B, D]
        next_obs = samples.next_observations
        rewards = samples.rewards.squeeze(-1)
        terms = samples.terminations.squeeze(-1)
        
        # Use the device of the data (already on GPU if full_gpu=True)
        device = obs.device
        
        # 1. Current Q-values
        values, d_adv, c_adv = self.q_net(obs)
        
        # Gather advantages for taken actions — pre-allocated to avoid list+cat overhead
        n_heads = (len(self.discrete_action_dims) if self.discrete_action_dims else 0) + self.continuous_action_dims
        q_heads = values.new_zeros(obs.shape[0], n_heads)
        col = 0
        idx = 0
        if self.discrete_action_dims:
            for i in range(len(self.discrete_action_dims)):
                q_heads[:, col] = d_adv[i].gather(1, actions[:, idx:idx+1].long()).squeeze(-1)
                idx += 1; col += 1
        if self.continuous_action_dims:
            for i in range(self.continuous_action_dims):
                val = actions[:, idx]
                bin_idx = torch.round((val - self._min_t[i]) / self._bin_widths[i]).long().clamp(0, self.n_c_action_bins - 1)
                q_heads[:, col] = c_adv[i].gather(1, bin_idx.unsqueeze(-1)).squeeze(-1)
                idx += 1; col += 1
        v = values.squeeze(-1)
        
        importance_raw = None
        if self.mix_type == "QMIX":
            # Merge V and A: feed per-head action-values Q_h = V(s) + A_h(s,a_h)
            # into the monotone mixer (standard QMIX over Q-values) instead of
            # mixing bare mean-centred advantages and adding a separate V.  The
            # latter is non-identifiable (a constant floats between V and the
            # mixer) and feeds the mixer symmetric-around-0 inputs that the
            # asymmetric leaky_relu cannot represent on the negative side, which
            # drives the bootstrapped-target divergence (see report H3).
            q_heads_full = q_heads + v.unsqueeze(-1)
            current_q_tot, _ = self.q_net.factorize_Q(q_heads_full, obs, with_grad=False)
            current_q_tot = current_q_tot.squeeze(-1)  # value already inside Q_h

            q_heads_detached = q_heads_full.detach().clone()
            q_heads_detached.requires_grad = True
            _, q_grads = self.q_net.factorize_Q(q_heads_detached, obs, with_grad=True)
            self.q_net.zero_grad()
            if q_grads is not None:
                importance_raw = (q_heads_detached.detach() * q_grads).cpu().numpy()
            else:
                importance_raw = np.zeros_like(q_heads_detached.detach().cpu().numpy())
        elif self.mix_type == "VDN":
            current_q_tot = q_heads.sum(dim=-1) + v
        else:
            # Independent heads: each head plus value should match global target
            # current_q_tot becomes [B, n_heads]
            current_q_tot = q_heads + v.unsqueeze(-1)
                
        # 2. Target Q-values
        with torch.no_grad():
            next_values, next_d_adv, next_c_adv = self.target_q_net(next_obs)

            nq_list = []
            if next_d_adv:
                for head in next_d_adv:
                    if self.dqn_type in (dqntype.Soft, dqntype.Munchausen):
                        lprobs = torch.log_softmax(head / self.entropy_loss_coef, dim=-1)
                        soft_nq = (lprobs.exp() * (head - self.entropy_loss_coef * lprobs)).sum(dim=-1)
                        nq_list.append(soft_nq.unsqueeze(-1))
                    else:
                        nq_list.append(head.max(dim=-1).values.unsqueeze(-1))
            if next_c_adv:
                for head in next_c_adv:
                    if self.dqn_type in (dqntype.Soft, dqntype.Munchausen):
                        lprobs = torch.log_softmax(head / self.entropy_loss_coef, dim=-1)
                        soft_nq = (lprobs.exp() * (head - self.entropy_loss_coef * lprobs)).sum(dim=-1)
                        nq_list.append(soft_nq.unsqueeze(-1))
                    else:
                        nq_list.append(head.max(dim=-1).values.unsqueeze(-1))

            nq_heads = torch.cat(nq_list, dim=-1)
            nv = next_values.squeeze(-1)

            if self.mix_type == "QMIX":
                # Merge V and A on the target side too: max_a Q_h = V(s') + max_a A_h
                # (V is constant across actions), then mix the per-head Q-values.
                nq_heads_full = nq_heads + nv.unsqueeze(-1)
                next_q_tot, _ = self.target_q_net.factorize_Q(nq_heads_full, next_obs)
                next_q_tot = next_q_tot.squeeze(-1)
            elif self.mix_type == "VDN":
                next_q_tot = nq_heads.sum(dim=-1) + nv
            else:
                # Independent: target is global reward plus max of LOCAL head Qs
                # next_q_tot is [B, n_heads]
                next_q_tot = nq_heads + nv.unsqueeze(-1)

            # rewards and terms are [B], target_q will be [B] (mix) or [B, n_heads] (nomix)
            target_q = rewards.unsqueeze(-1) + self.gamma * (1.0 - terms.unsqueeze(-1)) * next_q_tot if self.mix_type is None else rewards + self.gamma * (1.0 - terms) * next_q_tot

            # Munchausen reward augmentation: add α·τ·log π(a|s) to targets
            if self.dqn_type == dqntype.Munchausen:
                idx = 0
                if self.discrete_action_dims:
                    for i in range(len(self.discrete_action_dims)):
                        log_pi = Categorical(
                            logits=d_adv[i].detach() / self.entropy_loss_coef
                        ).log_prob(actions[:, idx].long())
                        
                        inc = self.entropy_loss_coef * self.munchausen * log_pi
                        if self.mix_type is None:
                            target_q[:, i] = target_q[:, i] + inc
                        else:
                            target_q = target_q + inc
                        idx += 1
                if self.continuous_action_dims:
                    for i in range(self.continuous_action_dims):
                        c_val = actions[:, idx]
                        bin_idx = torch.round(
                            (c_val - self._min_t[i]) / self._bin_widths[i]
                        ).long().clamp(0, self.n_c_action_bins - 1)
                        log_pi = torch.log_softmax(
                            c_adv[i].detach() / self.entropy_loss_coef, dim=-1
                        ).gather(dim=-1, index=bin_idx.unsqueeze(-1)).squeeze(-1)
                        
                        inc = self.entropy_loss_coef * self.munchausen * log_pi
                        if self.mix_type is None:
                            target_q[:, len(self.discrete_action_dims or []) + i] = target_q[:, len(self.discrete_action_dims or []) + i] + inc
                        else:
                            target_q = target_q + inc
                        idx += 1
            
        loss = F.mse_loss(current_q_tot, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update_interval == 0:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.lerp_(param.data, self.tau)
                
        res = {"rl_loss": loss.item()}
        if importance_raw is not None:
            res["importance_raw"] = importance_raw
        return res

    def _discretize_actions(self, continuous_actions: torch.Tensor) -> torch.Tensor:
        assert self.np_action_means is not None and self.np_action_ranges is not None, \
            "Cannot discretize continuous actions without action bounds (max_actions / min_actions)."
        if not torch.is_tensor(continuous_actions):
            continuous_actions = T(continuous_actions, device=self.device)
        res = []
        for i in range(self.continuous_action_dims):
            val = continuous_actions[..., i]
            means_i = torch.tensor(self.np_action_means[i], device=self.device, dtype=torch.float32)
            ranges_i = torch.tensor(self.np_action_ranges[i], device=self.device, dtype=torch.float32)
            d = torch.clamp(
                torch.round(((val - means_i) / ranges_i + 0.5) * (self.n_c_action_bins - 1)).long(),
                0, self.n_c_action_bins - 1,
            )
            res.append(d)
        return torch.stack(res, dim=-1)

    def _bc_cross_entropy_loss(self, disc_adv, cont_adv, disc_act, cont_act):
        discrete_loss = torch.zeros(1, device=self.device)
        continuous_loss = torch.zeros(1, device=self.device)
        if self.discrete_action_dims and disc_adv is not None and disc_act is not None:
            for i in range(len(self.discrete_action_dims)):
                discrete_loss = discrete_loss + nn.CrossEntropyLoss()(disc_adv[i], disc_act[:, i].long())
        if self.continuous_action_dims > 0 and cont_adv is not None and cont_act is not None:
            bin_indices = self._discretize_actions(cont_act)
            for i in range(self.continuous_action_dims):
                continuous_loss = continuous_loss + nn.CrossEntropyLoss()(cont_adv[i], bin_indices[:, i])
        return discrete_loss, continuous_loss

    def _reward_imitation_loss(self, disc_adv, cont_adv, disc_act, cont_act):
        discrete_loss = torch.zeros(1, device=self.device)
        continuous_loss = torch.zeros(1, device=self.device)
        if self.discrete_action_dims and disc_adv is not None and disc_act is not None:
            for i in range(len(self.discrete_action_dims)):
                best_q, best_a = torch.max(disc_adv[i], -1)
                mask = (best_a != disc_act[:, i].long()).float()
                discrete_loss = discrete_loss + nn.MSELoss(reduction="none")(best_q + mask, best_q.detach()).mean()
        if self.continuous_action_dims > 0 and cont_adv is not None and cont_act is not None:
            bin_indices = self._discretize_actions(cont_act)
            for i in range(self.continuous_action_dims):
                best_q, best_a = torch.max(cont_adv[i], -1)
                mask = (best_a != bin_indices[:, i]).float()
                continuous_loss = continuous_loss + nn.MSELoss(reduction="none")(best_q + mask, best_q.detach()).mean()
        return discrete_loss, continuous_loss

    def imitation_learn(self, observations, continuous_actions, discrete_actions, action_mask=None, debug=False):
        if self.eval_mode:
            return {"im_discrete_loss": 0, "im_continuous_loss": 0}
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device)
        if discrete_actions is not None and not torch.is_tensor(discrete_actions):
            discrete_actions = torch.tensor(discrete_actions, dtype=torch.long, device=self.device)
        if continuous_actions is not None and not torch.is_tensor(continuous_actions):
            continuous_actions = T(continuous_actions, device=self.device)

        values, disc_adv, cont_adv = self.q_net(observations)
        if self.imitation_type == "cross_entropy":
            dloss, closs = self._bc_cross_entropy_loss(disc_adv, cont_adv, discrete_actions, continuous_actions)
        else:
            dloss, closs = self._reward_imitation_loss(disc_adv, cont_adv, discrete_actions, continuous_actions)

        loss = dloss + closs
        if isinstance(loss, torch.Tensor) and loss.item() > 0:
            self.imitation_optimizer.zero_grad()
            loss.backward()
            self.imitation_optimizer.step()

        return {
            "im_discrete_loss": dloss.item() if isinstance(dloss, torch.Tensor) else 0,
            "im_continuous_loss": closs.item() if isinstance(closs, torch.Tensor) else 0,
        }

    def param_count(self) -> tuple[int, int]:
        actor_params = sum(p.numel() for p in self.q_net.parameters())
        return actor_params, actor_params

    def ego_actions(self, observations, action_mask=None):
        return self.train_actions(observations, action_mask=action_mask, epsilon=0.0)

    def expected_V(self, obs, legal_action=None):
        if not torch.is_tensor(obs):
            obs = T(obs, device=self.device)
        with torch.no_grad():
            values, d_adv, c_adv = self.q_net(obs)
            v = values.squeeze(-1)

            nq_list = []
            if d_adv:
                for head in d_adv:
                    nq_list.append(head.max(dim=-1).values.unsqueeze(-1))
            if c_adv:
                for head in c_adv:
                    nq_list.append(head.max(dim=-1).values.unsqueeze(-1))
            
            if not nq_list:
                return v

            nq_heads = torch.cat(nq_list, dim=-1)

            if self.mix_type == "QMIX":
                # Merge V and A: mix per-head Q_h = V + max_a A_h (value already inside)
                q_mix, _ = self.q_net.factorize_Q(nq_heads + v.unsqueeze(-1), obs)
                return q_mix.squeeze(-1)
            elif self.mix_type == "VDN":
                return v + nq_heads.sum(dim=-1)
            else:
                # For independent DQN, V is usually max(Q) of a single head or mean of max(Q)s.
                # Let's use the sum of max advantages + V for consistency with the nomix case in RL.
                # Actually, nomix in RL trains each head independently. 
                # Returning the mean max Q across heads is a reasonable estimate of V.
                return v + nq_heads.mean(dim=-1)

    def stable_greedy(self, obs, legal_action):
        acts = self.train_actions(obs, action_mask=legal_action, epsilon=0.0)
        adiscrete = torch.as_tensor(acts["discrete_actions"], device=self.device) if acts["discrete_actions"] is not None else None
        acontinuous = torch.as_tensor(acts["continuous_actions"], device=self.device) if acts["continuous_actions"] is not None else None
        return adiscrete, acontinuous

    def utility_function(self, observations, actions=None):
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device)
        values, d_adv, c_adv = self.q_net(observations)
        if actions is None:
            return values, d_adv, c_adv
        # If actions provided, return Q(s, a)
        q = values.squeeze(-1)
        # ... gather logic similar to reinforcement_learn ...
        return q

    def save(self, checkpoint_path):
        if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
        torch.save(self.q_net.state_dict(), os.path.join(checkpoint_path, "Q"))
        
    def load(self, checkpoint_path):
        self.q_net.load_state_dict(torch.load(os.path.join(checkpoint_path, "Q"), map_location=self.device))
        self.target_q_net.load_state_dict(self.q_net.state_dict())
