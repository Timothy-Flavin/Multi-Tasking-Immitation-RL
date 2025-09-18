# %%
import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Categorical
from .Agent import Agent
from .Agent import QS
from flexibuff import FlexiBatch
import os
import pickle
import warnings
import time

from enum import Enum


class dqntype(Enum):
    EGreedy = 0
    Soft = 1
    Munchausen = 2


# %%


class DQN(nn.Module, Agent):
    def __init__(
        self,
        obs_dim=10,
        discrete_action_dims=None,  # np.array([2]),
        continuous_action_dims: int = 0,  # 2,
        min_actions=None,  # np.array([-1,-1]),
        max_actions=None,  # ,np.array([1,1]),
        hidden_dims=[64, 64],  # first is obs dim if encoder provded
        head_hidden_dim=None,  # if None then no head hidden layer
        gamma=0.99,
        lr=3e-5,
        imitation_lr=1e-5,
        dueling=False,
        n_c_action_bins=5,
        munchausen=0.0,  # turns it into munchausen dqn
        entropy=0.0,  # turns it into soft-dqn
        activation="relu",
        orthogonal=False,
        init_eps=0.9,
        eps_decay_half_life=10000,
        device="cpu",
        eval_mode=False,
        name="DQN",
        clip_grad=1.0,
        load_from_checkpoint_path=None,
        encoder=None,
        conservative=False,
        imitation_type="cross_entropy",  # or "reward"
        mix_type: None | str = "None",  # None, VDN, QMIX
        wall_time=False,
        mix_dim=32,
    ):
        super(DQN, self).__init__()
        config = locals()
        config.pop("self")
        self.config = config

        self.wall_time = wall_time
        if mix_type is None or mix_type.lower() == "none":
            mix_type = None
        if mix_type is not None:
            if mix_type.lower() == "vdn":
                mix_type = "VDN"
            elif mix_type.lower() == "qmix":
                mix_type = "QMIX"
        self.mix_type = mix_type
        self.device = device
        self.clip_grad = clip_grad
        if load_from_checkpoint_path is not None:
            self.load(load_from_checkpoint_path)
            return
        self.eval_mode = eval_mode
        self.imitation_type = imitation_type
        self.entropy_loss_coef = entropy  # use soft Q learning entropy loss or not H(Q)
        self.dqn_type = dqntype.EGreedy
        if self.entropy_loss_coef > 0:
            self.dqn_type = dqntype.Soft
        if self.entropy_loss_coef > 0 and munchausen > 0:
            self.dqn_type = dqntype.Munchausen

        self.obs_dim = obs_dim  # size of observation
        self.discrete_action_dims = discrete_action_dims
        self.imitation_lr = imitation_lr
        # cardonality for each discrete action

        self.continuous_action_dims = continuous_action_dims
        # number of continuous actions

        self.name = name
        self.min_actions = min_actions  # min continuous action value
        self.max_actions = max_actions  # max continuous action value
        if self.max_actions is not None and self.min_actions is not None:
            self.np_action_ranges = self.max_actions - self.min_actions
            self.action_ranges = torch.from_numpy(self.np_action_ranges).to(device)
            self.np_action_means = (self.max_actions + self.min_actions) / 2
            self.action_means = torch.from_numpy(self.np_action_means).to(device)
        self.gamma = gamma
        self.lr = lr
        self.dueling = (
            dueling  # whether or not to learn True: V+Adv = Q or False: Adv = Q
        )
        self.n_c_action_bins = n_c_action_bins  # number of discrete action bins to discretize continuous actions
        self.munchausen = munchausen  # munchausen amount
        self.twin = False  # min(double q) to reduce bias
        self.init_eps = init_eps  # starting eps_greedy epsilon
        self.eps = self.init_eps
        self.eps_decay_half_life = (
            eps_decay_half_life  # eps cut in half every 'half_life' frames
        )
        self.step = 0
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.orthogonal = orthogonal
        # print("before qs")
        self.Q1 = QS(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dims,
            discrete_action_dims=discrete_action_dims,
            hidden_dims=hidden_dims,
            activation=activation,
            orthogonal=orthogonal,
            dueling=dueling,
            n_c_action_bins=n_c_action_bins,
            device=device,
            encoder=encoder,  # pass encoder if using one for observations (like in visual DQN)
            head_hidden_dims=(
                np.copy(np.array(head_hidden_dim))
                if head_hidden_dim is not None
                else None
            ),  # if None then no head hidden layer
            QMIX=self.mix_type == "QMIX",
            QMIX_hidden_dim=mix_dim,
        )
        self.encoder = self.Q1.encoder
        self.Q1.to(device)

        self.Q2 = QS(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dims,
            discrete_action_dims=discrete_action_dims,
            hidden_dims=hidden_dims,
            activation=activation,
            orthogonal=orthogonal,
            dueling=dueling,
            n_c_action_bins=n_c_action_bins,
            device=device,
            encoder=encoder,  # pass encoder if using one for observations (like in visual DQN)
            head_hidden_dims=(
                np.copy(np.array(head_hidden_dim))
                if head_hidden_dim is not None
                else None
            ),  # if None then no head hidden layer
            QMIX=self.mix_type == "QMIX",
            QMIX_hidden_dim=mix_dim,
        )
        self.head_hidden_dims = head_hidden_dim
        self.Q2.to(self.device)
        with torch.no_grad():
            self.Q2.load_state_dict(self.Q1.state_dict())
        for param in self.Q2.parameters():
            param.requires_grad = False

        self.update_num = 0
        self.conservative = conservative
        # self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=0.1)
        self.to(device)
        self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr)

        self.has_discrete = (
            self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0
        )
        self.has_continuous = self.continuous_action_dims > 0
        # These can be saved to remake the same DQN
        # TODO: check that this is suffucuent
        self.attrs = [
            "step",
            "entropy_loss_coef",
            "munchausen",
            "discrete_action_dims",
            "continuous_action_dims",
            "min_actions",
            "max_actions",
            "gamma",
            "lr",
            "dueling",
            "n_c_action_bins",
            "init_eps",
            "eps_decay_half_life",
            "device",
            "eval_mode",
            "hidden_dims",
            "activation",
        ]

    def _cont_from_q(self, cont_act):
        return (
            torch.argmax(cont_act, dim=-1) / (self.n_c_action_bins - 1) - 0.5
        ) * self.action_ranges + self.action_means

    def _cont_from_soft_q(self, cont_act):
        # print(Categorical(logits=cont_act / self.entropy_loss_coef).probs)
        return (
            Categorical(logits=cont_act / self.entropy_loss_coef).sample()
            / (self.n_c_action_bins - 1)
            - 0.5
        ) * self.action_ranges + self.action_means

    def _discretize_actions(self, continuous_actions):
        # print(continuous_actions.shape)
        return torch.clamp(  # inverse of _cont_from_q
            torch.round(
                ((continuous_actions - self.action_means) / self.action_ranges + 0.5)
                * (self.n_c_action_bins - 1)
            ).to(torch.int64),
            0,
            self.n_c_action_bins - 1,
        )

    def _e_greedy_train_action(
        self, observations, action_mask=None, step=False, debug=False
    ):
        disc_act, cont_act = None, None
        if self.init_eps > 0.0:
            self.eps = self.init_eps * (
                1 - self.step / (self.step + self.eps_decay_half_life)
            )
        value = 0
        if self.init_eps > 0.0 and np.random.rand() < self.eps:
            # print("Random action")
            if (
                self.discrete_action_dims is not None
                and len(self.discrete_action_dims) > 0
            ):
                disc_act = np.zeros(
                    shape=len(self.discrete_action_dims), dtype=np.int32
                )
                for i in range(len(self.discrete_action_dims)):
                    disc_act[i] = np.random.randint(0, self.discrete_action_dims[i])

            if self.continuous_action_dims > 0:
                cont_act = (
                    np.random.rand(self.continuous_action_dims) - 0.5
                ) * self.np_action_ranges + self.np_action_means
            # print(disc_act)
        else:
            # print("Not random action")
            with torch.no_grad():
                # print("Getting value from Q1 for soft action selection")
                value, disc_act, cont_act = self.Q1(observations, action_mask)
                # print("done with that")
                # select actions from q function
                # print(value, disc_act, cont_act)
                if (
                    self.discrete_action_dims is not None
                    and len(self.discrete_action_dims) > 0
                ):
                    d_act = np.zeros(len(disc_act), dtype=np.int32)
                    for i, da in enumerate(disc_act):
                        d_act[i] = torch.argmax(da).detach().cpu().item()
                    disc_act = d_act
                if self.continuous_action_dims > 0:
                    if debug:
                        print(
                            f"  cont act {cont_act}, argmax: {torch.argmax(cont_act,dim=-1).detach().cpu()}"
                        )
                        print(
                            f"  Trying to store this in actions {((torch.argmax(cont_act,dim=-1)/ (self.n_c_action_bins - 1) -0.5)* self.action_ranges+ self.action_means)} calculated from da: {cont_act} with ranges: {self.action_ranges} and means: {self.action_means}"
                        )
                    cont_act = self._cont_from_q(cont_act).cpu().numpy()
        return disc_act, cont_act

    def _soft_train_action(self, observations, action_mask, step, debug):
        disc_act, cont_act = None, None
        with torch.no_grad():
            value, disc_act, cont_act = self.Q1(observations, action_mask)
            # print("Done with that")
            if (
                self.discrete_action_dims is not None
                and len(self.discrete_action_dims) > 0
            ):
                dact = np.zeros(len(disc_act), dtype=np.int64)
                for i, da in enumerate(disc_act):
                    dact[i] = Categorical(logits=da).sample().cpu().item()
                disc_act = dact  # had to store da temporarily to keep using disc_act
            if self.continuous_action_dims > 0:
                if debug:
                    print(
                        f"  cont act {cont_act}, argmax: {torch.argmax(cont_act,dim=-1).detach().cpu()}"
                    )
                    print(
                        f"  Trying to store this in actions {((torch.argmax(cont_act,dim=-1)/ (self.n_c_action_bins - 1) -0.5)* self.action_ranges+ self.action_means)} calculated from da: {cont_act} with ranges: {self.action_ranges} and means: {self.action_means}"
                    )
                cont_act = self._cont_from_soft_q(cont_act).cpu().numpy()
        return disc_act, cont_act

    def train_actions(self, observations, action_mask=None, step=False, debug=False):
        t = 0
        if self.wall_time:
            t = time.time()
        disc_act, cont_act = self._e_greedy_train_action(
            observations, action_mask, step, debug
        )
        self.step += int(step)
        if self.wall_time:
            t = time.time() - t
        return {
            "discrete_actions": disc_act,
            "continuous_actions": cont_act,
            "act_time": t,
        }

    def ego_actions(self, observations, action_mask=None):
        return {"discrete_actions": 0, "continuous_actions": 0, "action_time": 0}

    def stable_greedy(self, obs, legal_action):
        with torch.no_grad():
            values, disc_advantages, cont_advantages = self.Q2(obs)
            dact = None
            cact = None
            if self.has_discrete:
                dact = []
                for dh in disc_advantages:
                    dact.append(torch.argmax(dh, dim=-1).unsqueeze(0))
                torch.cat(dact, dim=0)
            if self.has_continuous:
                cact = torch.argmax(cont_advantages, dim=-1)

        return dact, cact

    def _bc_cross_entropy_loss(self, disc_adv, cont_adv, disc_act, cont_act):
        discrete_loss = 0
        continuous_loss = 0
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            for i in range(len(self.discrete_action_dims)):
                discrete_loss += nn.CrossEntropyLoss()(
                    disc_adv[i], disc_act[:, i]
                )  # for discrete action 1

        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            continuous_actions = self._discretize_actions(cont_act)
            # print(continuous_actions.shape)
            for i in range(self.continuous_action_dims):
                continuous_loss += nn.CrossEntropyLoss()(
                    cont_adv[:, i], continuous_actions[:, i]
                )

        return discrete_loss, continuous_loss

    def _reward_imitation_loss(self, disc_adv, cont_adv, disc_act, cont_act):
        discrete_loss = 0
        continuous_loss = 0
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            for i in range(len(self.discrete_action_dims)):
                best_q, best_a = torch.max(disc_adv[i], -1)
                mask = best_a != disc_act[:, i]
                discrete_loss += nn.MSELoss(reduction="none")(
                    best_q + mask, best_q.detach()
                ).mean()

        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            continuous_actions = self._discretize_actions(cont_act)
            # print(continuous_actions)
            for i in range(self.continuous_action_dims):
                best_q, best_a = torch.max(cont_adv[:, i], -1)
                mask = best_a != continuous_actions[:, i]
                continuous_loss += nn.MSELoss(reduction="none")(
                    best_q + mask, best_q.detach()
                ).mean()
        return discrete_loss, continuous_loss

    def imitation_learn(
        self,
        observations,
        continuous_actions,
        discrete_actions,
        action_mask=None,
        debug=False,
    ):
        t = 0
        if self.wall_time:
            t = time.time()
        values, disc_adv, cont_adv = self.Q1(observations)
        if self.eval_mode:
            return {"im_discrete_loss": 0, "im_continuous_loss": 0}
        else:
            dloss, closs = torch.zeros(1, device=self.device), torch.zeros(
                1, device=self.device
            )
            # print(
            #     f" imitation shapes: d_adv {disc_adv.shape if disc_adv is not None else None}, c_adv {cont_adv.shape if cont_adv is not None else None}, dact {discrete_actions.shape if discrete_actions is not None else None} cact {continuous_actions.shape if continuous_actions is not None else None}"
            # )
            if self.imitation_type == "cross_entropy":
                dloss, closs = self._bc_cross_entropy_loss(
                    disc_adv, cont_adv, discrete_actions, continuous_actions
                )
            else:
                dloss, closs = self._reward_imitation_loss(
                    disc_adv, cont_adv, discrete_actions, continuous_actions
                )
            loss = dloss + closs
            assert isinstance(loss, torch.Tensor), "Loss needs to be a tensor"
            if loss == 0:
                warnings.warn(
                    "Loss is 0, not updating. Most likely due to continuous and discrete actions being None,0 respectively"
                )
                return {"im_discrete_loss": 0, "im_continuous_loss": 0}
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad is not None and self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.clip_grad,
                    error_if_nonfinite=True,
                    foreach=True,
                )
            self.optimizer.step()
            if dloss != 0:
                dloss = dloss.item()
            if closs != 0:
                closs = closs.item()

            if self.wall_time:
                t = time.time() - t
            return {"im_discrete_loss": dloss, "im_continuous_loss": closs, "time": t}

    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action=None, debug=False):
        with torch.no_grad():
            v, dac, cac = self.Q1(obs, legal_action)
            if self.dueling:
                return v.squeeze(-1)
            else:
                evs = []
                if (
                    self.discrete_action_dims is not None
                    and len(self.discrete_action_dims) > 0
                ):
                    for h in dac:
                        evs.append(torch.max(h, dim=-1, keepdim=True)[0])
                if self.continuous_action_dims > 0:
                    evs.append(torch.max(cac, dim=-1)[0])
                return torch.cat(evs, dim=-1).mean(dim=-1)

    def cql_loss(self, disc_adv, cont_adv, disc_act, cont_act):
        """Computes the CQL loss for a batch of Q-values and actions."""
        cql_loss = 0
        if self.discrete_action_dims is not None:
            for i in range(len(self.discrete_action_dims)):
                logsumexp = torch.logsumexp(disc_adv[i], dim=-1, keepdim=True)
                q_a = disc_adv[i].gather(1, disc_act[:, i].unsqueeze(-1))
                cql_loss += (logsumexp - q_a).mean()
        for i in range(self.continuous_action_dims):
            logsumexp = torch.logsumexp(cont_adv[i], dim=-1, keepdim=True)
            q_a = cont_adv[i].gather(1, cont_act[:, i])
            cql_loss += (logsumexp - q_a).mean()

        return cql_loss

    def _discrete_next_q(self, values, advantages, action_dim=None, debug=True):
        assert (
            action_dim is not None
        ), "Cant be jagged=True with no action dim passed to _target()"
        Q_ = torch.zeros(
            size=(advantages[0].shape[0], len(action_dim)),
            device=self.device,
            dtype=torch.float32,
        )
        for i in range(len(action_dim)):
            # Treat actions as probabalistic if using soft Q or m-dqn
            if self.dqn_type == dqntype.Munchausen or self.dqn_type == dqntype.Soft:
                lprobs = torch.log_softmax(
                    advantages[i] / self.entropy_loss_coef, dim=-1
                )
                probs = torch.exp(lprobs)
                Q_[:, i] = torch.sum(
                    probs * (advantages[i] - self.entropy_loss_coef * lprobs),
                    dim=-1,
                )
            else:
                Q_[:, i] = torch.max(advantages[i], dim=-1).values
        if self.mix_type == "VDN":
            Q_ = Q_.sum(dim=-1)
        elif self.mix_type is None:
            # print(f"dQ_: {Q_.shape}, values: {values.shape}")
            Q_ = Q_ + values
        return Q_

    def _continuous_next_q(self, values, advantages, debug=False):
        if self.dqn_type == dqntype.Munchausen or self.dqn_type == dqntype.Soft:
            lprobs = torch.log_softmax(advantages / self.entropy_loss_coef, dim=-1)
            probs = torch.exp(lprobs)
            Q_ = torch.sum(
                probs * (advantages - self.entropy_loss_coef * lprobs), dim=-1
            )
        else:
            Q_ = torch.max(advantages, dim=-1).values
        if self.mix_type is None:
            # print(f"cQ_: {Q_.shape}, values: {values.shape}")
            Q_ = Q_ + values
        elif self.mix_type == "VDN":
            Q_ = Q_.sum(dim=-1)
        return Q_

    # torch no grad called in reinfrocement learn so no need here
    def _target(
        self,
        values,
        disc_advantages,
        cont_advantages,
        rewards,
        terminated,
        action_dim,
        debug=True,
        state=None,
    ):
        disc_targets, cont_targets, combined_targets = 0, 0, 0
        if not self.dueling:
            values = 0
        dQ_ = (
            self._discrete_next_q(values, disc_advantages, action_dim, debug)
            if self.has_discrete
            else 0
        )
        cQ_ = (
            self._continuous_next_q(values, cont_advantages, debug)
            if self.has_continuous
            else 0
        )

        if self.mix_type is None:
            if self.has_discrete:
                disc_targets = rewards.unsqueeze(-1) + (
                    self.gamma * (1 - terminated.unsqueeze(-1))
                ) * (dQ_)

                # print(
                #    f"In targ: rew: {rewards[0:5]}, terminated: {(1 - terminated.unsqueeze(-1))[0:5]}, dQ_: {dQ_[0:5]} values: {values[0:5]}"
                # )
            else:
                disc_targets = 0
            if self.has_continuous:
                cont_targets = rewards.unsqueeze(-1) + (
                    self.gamma * (1 - terminated.unsqueeze(-1))
                ) * (cQ_)
            else:
                cont_targets = 0
        else:
            vals = values.squeeze(-1) if isinstance(values, torch.Tensor) else 0
            if self.mix_type == "VDN":
                Q_ = dQ_ + cQ_ + vals
            elif self.mix_type == "QMIX":
                qlist = []
                if self.has_discrete:
                    qlist.append(dQ_)
                if self.has_continuous:
                    qlist.append(cQ_)
                Q_ = (
                    self.Q2.factorize_Q(torch.cat(qlist, dim=1), state)[0].squeeze(-1)
                    + vals
                )
            else:
                raise Exception("Mix type needs to be None VDN or QMIX")
            # print(
            #    f"combined shapes: {rewards.shape} 1-term: {(1-terminated).shape} Q_.shape: {Q_.shape}"
            # )
            combined_targets = rewards + (self.gamma * (1 - terminated)) * Q_
        return disc_targets, cont_targets, combined_targets

    def reinforcement_learn(
        self, batch: FlexiBatch, agent_num=0, critic_only=False, debug=False
    ):
        t = 0
        if self.wall_time:
            t = time.time()
        self.update_num += 1
        if self.eval_mode:
            return {"rl_loss": 0, "rl_time": 0}

        continuous_actions = None
        discrete_actions = None
        if self.discrete_action_dims is not None:
            discrete_actions = batch.discrete_actions[agent_num]  # type: ignore
        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            continuous_actions = self._discretize_actions(
                batch.continuous_actions[agent_num]  # type: ignore
            )
        discrete_target = 0  # torch.zeros(1, device=self.device)
        continuous_target = 0  # torch.zeros(1, device=self.device)
        values, disc_adv, cont_adv = self.Q1(batch.obs[agent_num])
        with torch.no_grad():
            next_values, next_disc_adv, next_cont_adv = self.Q2(batch.obs_[agent_num])
            discrete_target, continuous_target, combined_target = self._target(
                values=next_values,  # next_values,
                disc_advantages=next_disc_adv,
                cont_advantages=next_cont_adv,
                rewards=batch.global_rewards,
                terminated=batch.terminated,
                action_dim=self.discrete_action_dims,
                state=batch.obs_[agent_num],
                debug=debug,
            )
            if (
                self.discrete_action_dims is not None
                and len(self.discrete_action_dims) > 0
            ):
                assert (
                    discrete_actions is not None
                ), "Cant learn on discrete actions if they are None"
                if self.dqn_type == dqntype.Munchausen:
                    for i in range(len(self.discrete_action_dims)):
                        munchausen_reward = (
                            self.entropy_loss_coef
                            * self.munchausen
                            * (
                                Categorical(
                                    logits=disc_adv[i] / self.entropy_loss_coef
                                ).log_prob(discrete_actions[:, i])
                            )
                        )
                        if self.mix_type is None:
                            assert isinstance(
                                discrete_target, torch.Tensor
                            ), "If no mixing then continuous target needs to be tensor"
                            discrete_target[:, i] += munchausen_reward
                        else:
                            assert isinstance(
                                combined_target, torch.Tensor
                            ), "If mixing is enabled then combined target needs to be tensor"
                            combined_target += munchausen_reward
            if (
                self.continuous_action_dims is not None
                and self.continuous_action_dims > 0
            ):
                assert (
                    continuous_actions is not None
                ), "Cant learn on continuous actions if they are None"
                if self.dqn_type == dqntype.Munchausen:
                    munchausen_reward = (
                        self.entropy_loss_coef
                        * self.munchausen
                        * torch.log_softmax(cont_adv / self.entropy_loss_coef, dim=-1)
                        .gather(dim=-1, index=continuous_actions.unsqueeze(-1))
                        .squeeze(-1)
                    )
                    if self.mix_type is None:
                        assert isinstance(
                            continuous_target, torch.Tensor
                        ), "If no mixing then continuous target needs to be tensor"
                        continuous_target += munchausen_reward
                    else:
                        assert isinstance(
                            combined_target, torch.Tensor
                        ), "If mixing is enabled then combined target needs to be tensor"
                        combined_target += munchausen_reward.sum(-1)

        cQ = 0
        if self.has_continuous:
            assert (
                continuous_actions is not None
            ), "Cant do continuous action update if actions are None"
            cQ = torch.gather(
                input=cont_adv,
                dim=-1,
                index=continuous_actions.unsqueeze(-1),
            ).squeeze(
                -1
            )  # + (values.squeeze(-1) if self.dueling else 0)
            if self.mix_type == "VDN":
                cQ = cQ.sum(-1)

        dQ = 0
        if self.has_discrete:
            assert (
                discrete_actions is not None and self.discrete_action_dims is not None
            ), "Cant do discrete update if discrete actions or action dims are None"
            dQ = torch.zeros(
                size=(batch.global_rewards.shape[0], len(self.discrete_action_dims)),
                device=self.device,
                dtype=torch.float32,
            )
            for d in range(len(self.discrete_action_dims)):
                dQ[:, d] = torch.gather(
                    disc_adv[d],
                    dim=-1,
                    index=discrete_actions[:, d].unsqueeze(-1),
                ).squeeze(-1)
            if self.mix_type == "VDN":
                dQ = dQ.sum(-1)
        loss = 0
        if self.mix_type is None:
            dloss = 0
            closs = 0
            if self.has_discrete:
                dloss = (((dQ + values) - discrete_target) ** 2).mean()
            if self.has_continuous:
                closs = (
                    ((cQ + values - continuous_target) ** 2).mean()
                    if self.has_continuous
                    else 0
                )
            loss = closs + dloss
        else:
            v = values.squeeze(-1) if isinstance(values, torch.Tensor) else 0.0
            Q = None
            if self.mix_type == "VDN":
                Q = dQ + cQ + v
            elif self.mix_type == "QMIX":
                qs = []
                if self.has_discrete:
                    qs.append(dQ)
                if self.has_continuous:
                    qs.append(cQ)

                Q = (
                    self.Q1.factorize_Q(torch.cat(qs, dim=-1), batch.obs[agent_num])[
                        0
                    ].squeeze(-1)
                    + v
                )

            assert isinstance(
                Q, torch.Tensor
            ), "Can't learn when the current q values don't exist"
            loss = ((Q - combined_target) ** 2).mean()

        assert isinstance(loss, torch.Tensor)
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.clip_grad,
                error_if_nonfinite=True,
                foreach=True,
            )
        for name, param in self.Q2.named_parameters():
            if param.grad is not None:
                print(f"WARNING: Q2 param {name} has non-zero grad!")
        self.optimizer.step()
        tau = 0.005  # A typical value
        with torch.no_grad():
            for target_param, online_param in zip(
                self.Q2.parameters(), self.Q1.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )
        # input("\n\nNew iter?")

        l_ = loss.item()
        if self.wall_time:
            t = time.time() - t
        return {
            "rl_loss": l_,
            "rl_time": t,
        }

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
        torch.save(self.Q1.state_dict(), checkpoint_path + "/Q1")
        for i in range(len(self.attrs)):
            self._dump_attr(
                self.__dict__[self.attrs[i]], checkpoint_path + f"/{self.attrs[i]}"
            )

    def load(self, checkpoint_path):
        if checkpoint_path is None:
            checkpoint_path = "./" + self.name + "/"
        if not os.path.exists(checkpoint_path):
            return None
        for i in range(len(self.attrs)):
            self.__dict__[self.attrs[i]] = self._load_attr(
                checkpoint_path + f"/{self.attrs[i]}"
            )

        self.dqn_type = dqntype.EGreedy
        if self.entropy_loss_coef > 0:
            self.dqn_type = dqntype.Soft
        if self.entropy_loss_coef > 0 and self.munchausen > 0:
            self.dqn_type = dqntype.Munchausen
        if self.max_actions is not None:
            self.np_action_ranges = self.max_actions - self.min_actions
            self.action_ranges = torch.from_numpy(self.np_action_ranges).to(self.device)
            self.np_action_means = (self.max_actions + self.min_actions) / 2
            self.action_means = torch.from_numpy(self.np_action_means).to(self.device)

        if self.Q1 is None:
            self.Q1 = QS(
                obs_dim=self.obs_dim,
                continuous_action_dim=self.continuous_action_dims,
                discrete_action_dims=self.discrete_action_dims,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                orthogonal=self.orthogonal,
                dueling=self.dueling,
                n_c_action_bins=self.n_c_action_bins,
                device=self.device,
                QMIX=self.mix_type == "QMIX",
            )
            self.Q2 = QS(
                obs_dim=self.obs_dim,
                continuous_action_dim=self.continuous_action_dims,
                discrete_action_dims=self.discrete_action_dims,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                orthogonal=self.orthogonal,
                dueling=self.dueling,
                n_c_action_bins=self.n_c_action_bins,
                device=self.device,
                QMIX=self.mix_type == "QMIX",
            )

        self.Q1.load_state_dict(torch.load(checkpoint_path + "/Q1", weights_only=True))
        self.Q1.to(self.device)
        with torch.no_grad():
            self.Q2.load_state_dict(self.Q1.state_dict())
            self.Q2.to(self.device)

        self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.to(self.device)

        return None

    def __str__(self):
        st = ""
        for i in self.__dict__.keys():
            st += f"{i}: {self.__dict__[i]}\n"
        return st

    def param_count(self):
        total_params = sum(p.numel() for p in self.Q1.parameters()) + sum(
            p.numel() for p in self.Q2.parameters()
        )
        exec_params = sum(p.numel() for p in self.Q1.parameters())
        if self.mix_type == "QMIX":
            assert (
                self.Q1.mixing_network is not None
            ), "If we are a q mixer then our q network needs a mixer"
            exec_params -= sum(p.numel() for p in self.Q1.mixing_network.parameters())

        return total_params, exec_params


# %%


if __name__ == "__main__":

    # %%
    obs_dim = 3
    continuous_action_dim = 2

    obs = np.random.rand(obs_dim).astype(np.float32)
    obs_ = np.random.rand(obs_dim).astype(np.float32)
    obs_batch = np.random.rand(14, obs_dim).astype(np.float32)
    obs_batch_ = obs_batch + 0.1

    dacs = np.stack(
        (np.random.randint(0, 4, size=(14)), np.random.randint(0, 5, size=(14))),
        axis=-1,
    )

    mem = FlexiBatch(
        registered_vals={
            "obs": np.array([obs_batch]),
            "obs_": np.array([obs_batch_]),
            "continuous_actions": np.array([np.random.rand(14, 2).astype(np.float32)]),
            "discrete_actions": np.array([dacs], dtype=np.int64),
            "global_rewards": np.random.rand(14).astype(np.float32),
        },
        terminated=np.random.randint(0, 2, size=14),
    )
    mem.to_torch("cuda:0")

    # print(f"expected v: {agent.expected_V(obs, legal_action=None)}")
    # exit()

    # %%
    agent = DQN(
        obs_dim=obs_dim,
        continuous_action_dims=continuous_action_dim,
        max_actions=np.array([1, 2]),
        min_actions=np.array([0, 0]),
        discrete_action_dims=[4, 5],
        hidden_dims=[32, 32],
        device="cuda:0",
        lr=0.001,
        activation="relu",
        dueling=True,
    )

    # %%

    # %%
    d_acts, c_acts, d_log, c_log, _1, _ = agent.train_actions(
        obs, step=True, debug=True
    )
    print(f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}")
    aloss, closs = agent.reinforcement_learn(mem, 0, critic_only=False, debug=True)
    print("Finished Testing")

    # %%

    agent = DQN(
        obs_dim=obs_dim,
        continuous_action_dims=continuous_action_dim,
        max_actions=np.array([1, 2]),
        min_actions=np.array([0, 0]),
        discrete_action_dims=[4, 5],
        hidden_dims=[64, 64],
        device="cuda:0",
        lr=3e-4,
        activation="relu",
        entropy=0.1,
    )
    d_acts, c_acts, d_log, c_log, _1, _ = agent.train_actions(
        obs, step=True, debug=True
    )
    print(f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}")
    aloss, closs = agent.reinforcement_learn(mem, 0, critic_only=False, debug=True)
    print("Finished Testing")

    agent = DQN(
        obs_dim=obs_dim,
        continuous_action_dims=continuous_action_dim,
        max_actions=np.array([1, 2]),
        min_actions=np.array([0, 0]),
        discrete_action_dims=[4, 5],
        hidden_dims=[32, 32],
        device="cuda:0",
        lr=0.001,
        activation="relu",
        entropy=0.1,
        munchausen=0.5,
    )
    d_acts, c_acts, d_log, c_log, _1, _ = agent.train_actions(
        obs, step=True, debug=True
    )
    print(f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}")
    aloss, closs = agent.reinforcement_learn(mem, 0, critic_only=False, debug=True)
    print("Finished Testing")

# %%
