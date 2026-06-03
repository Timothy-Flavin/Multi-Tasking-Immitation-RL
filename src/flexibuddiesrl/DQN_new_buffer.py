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
        
        self._get_torch_params(encoder, hidden_dims, head_hidden_dims, activation, orthogonal)
        
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
        
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device)
        
        is_3d = (len(observations.shape) == 3)
        if is_3d:
            n_agents, n_envs, obs_dim = observations.shape
            obs_flat = observations.reshape(-1, obs_dim)
        else:
            obs_flat = observations
            
        with torch.no_grad():
            values, d_adv, c_adv = self.q_net(obs_flat)
            
            d_actions = []
            if d_adv is not None:
                for head in d_adv:
                    if random.random() < epsilon and not self.eval_mode:
                        d_actions.append(torch.randint(0, head.shape[-1], (obs_flat.shape[0],), device=self.device))
                    else:
                        d_actions.append(head.argmax(dim=-1))
                d_actions = torch.stack(d_actions, dim=-1)
            else:
                d_actions = None
                
            c_actions = []
            if c_adv is not None:
                for head in c_adv:
                    if random.random() < epsilon and not self.eval_mode:
                        c_actions.append(torch.randint(0, head.shape[-1], (obs_flat.shape[0],), device=self.device))
                    else:
                        c_actions.append(head.argmax(dim=-1))
                c_actions = torch.stack(c_actions, dim=-1)
                
                # Unflatten and convert to numpy for continuous processing
                if is_3d:
                    c_actions_np = c_actions.reshape(n_agents, n_envs, -1).cpu().numpy()
                else:
                    c_actions_np = c_actions.cpu().numpy()

                # Convert bin indices back to continuous values
                c_actions_cont = []
                for i in range(self.continuous_action_dims):
                    bin_indices = c_actions_np[..., i]
                    bin_width = (self.np_max_actions[i] - self.np_min_actions[i]) / (self.n_c_action_bins - 1)
                    vals = self.np_min_actions[i] + bin_indices * bin_width
                    c_actions_cont.append(vals)
                c_actions_cont = np.stack(c_actions_cont, axis=-1)
            else:
                c_actions_cont = None

            # Unflatten results if input was 3D
            if is_3d:
                if d_actions is not None:
                    d_actions = d_actions.reshape(n_agents, n_envs, -1)
                if values is not None:
                    values = values.reshape(n_agents, n_envs, -1)

        return {
            "discrete_actions": d_actions.cpu().numpy() if d_actions is not None else None,
            "continuous_actions": c_actions_cont if c_actions_cont is not None else None,
            "values": values.cpu().numpy() if values is not None else None
        }

    def reinforcement_learn(self, samples: ReplayBufferSamples, agent_num=0):
        if self.eval_mode: return {}
        
        obs = samples.observations
        actions = samples.actions # [B, D]
        next_obs = samples.next_observations
        rewards = samples.rewards.squeeze(-1)
        terms = samples.terminations.squeeze(-1)
        
        # Use the device of the data (already on GPU if full_gpu=True)
        device = obs.device
        
        # 1. Current Q-values
        values, d_adv, c_adv = self.q_net(obs)
        
        # Gather advantages for taken actions
        q_list = []
        idx = 0
        if self.discrete_action_dims:
            for i in range(len(self.discrete_action_dims)):
                q_list.append(d_adv[i].gather(1, actions[:, idx:idx+1].long()))
                idx += 1
        if self.continuous_action_dims:
            for i in range(self.continuous_action_dims):
                val = actions[:, idx]
                bin_width = (self.np_max_actions[i] - self.np_min_actions[i]) / (self.n_c_action_bins - 1)
                bin_idx = torch.round((val - self.np_min_actions[i]) / bin_width).long().clamp(0, self.n_c_action_bins-1)
                q_list.append(c_adv[i].gather(1, bin_idx.unsqueeze(-1)))
                idx += 1
        
        q_heads = torch.cat(q_list, dim=-1) # [B, n_heads]
        v = values.squeeze(-1)
        
        importance_raw = None
        if self.mix_type == "QMIX":
            # Compute actual Q_tot attached to graph for training
            current_q_tot, _ = self.q_net.factorize_Q(q_heads, obs, with_grad=False)
            current_q_tot = current_q_tot.squeeze(-1) + v
            
            # Compute gradients for importance without affecting main graph training
            q_heads_detached = q_heads.detach().clone()
            q_heads_detached.requires_grad = True
            _, q_grads = self.q_net.factorize_Q(q_heads_detached, obs, with_grad=True)
            self.q_net.zero_grad() # Clear any hypernetwork grads from the detached backward
            if q_grads is not None:
                importance_raw = (q_heads.detach() * q_grads).cpu().numpy()
            else:
                importance_raw = np.zeros_like(q_heads.detach().cpu().numpy())
        elif self.mix_type == "VDN":
            current_q_tot = q_heads.sum(dim=-1) + v
        else:
            # Default to summing advantages if no mix_type provided but multiple heads exist
            current_q_tot = q_heads.sum(dim=-1) + v
                
        # 2. Target Q-values
        with torch.no_grad():
            next_values, next_d_adv, next_c_adv = self.target_q_net(next_obs)
            
            nq_list = []
            if next_d_adv:
                for head in next_d_adv:
                    nq_list.append(head.max(dim=-1).values.unsqueeze(-1))
            if next_c_adv:
                for head in next_c_adv:
                    nq_list.append(head.max(dim=-1).values.unsqueeze(-1))
            
            nq_heads = torch.cat(nq_list, dim=-1)
            nv = next_values.squeeze(-1)
            
            if self.mix_type == "QMIX":
                next_q_tot, _ = self.target_q_net.factorize_Q(nq_heads, next_obs)
                next_q_tot = next_q_tot.squeeze(-1) + nv
            elif self.mix_type == "VDN":
                next_q_tot = nq_heads.sum(dim=-1) + nv
            else:
                next_q_tot = nq_heads.sum(dim=-1) + nv
                    
            target_q = rewards + self.gamma * (1.0 - terms) * next_q_tot
            
        loss = F.mse_loss(current_q_tot, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.target_update_interval == 0:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                
        res = {"rl_loss": loss.item()}
        if importance_raw is not None:
            res["importance_raw"] = importance_raw
        return res

    def imitation_learn(self, observations, continuous_actions, discrete_actions, action_mask=None, debug=False):
        return {"rl_loss": 0}

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
            # For DQN, V is usually mean of Q or max Q. Original Agent.py often does this.
            # Let's use max Q as V.
            v = values.squeeze(-1)
            if d_adv:
                for head in d_adv:
                    v = v + head.max(dim=-1).values
            if c_adv:
                for head in c_adv:
                    v = v + head.max(dim=-1).values
            return v

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
