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
        value_per_head=False,
    ):
        super(DQN, self).__init__()
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
            value_per_head=value_per_head,
        )

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
            value_per_head=value_per_head,
        )
        self.Q2.to(self.device)
        with torch.no_grad():
            self.Q2.load_state_dict(self.Q1.state_dict())

        self.update_num = 0
        self.conservative = conservative
        # self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=0.1)
        self.to(device)

        # for i in range(50):
        #     obs = torch.rand(size=(128, obs_dim), device=self.device) - 0.5
        #     values, disc_yhat, cont_yhat = self.Q1(obs)
        #     print(
        #         f"vs: {values.shape if isinstance(values, torch.Tensor) else 0.0} dy: {[len(disc_yhat),disc_yhat[0].shape] if disc_yhat is not None else None} cy: {cont_yhat.shape if cont_yhat is not None else None}"
        #     )
        #     targ = torch.rand_like(yhat) / 10
        #     loss = ((yhat - targ) ** 2).mean()
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #     print(loss.item())
        self.optimizer = torch.optim.Adam(self.Q1.parameters(), lr=lr)
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
        # if len(observations.shape) == 1:
        #    observations = np.expand_dims(observations, axis=0)
        # if self.dqn_type == dqntype.Munchausen or self.dqn_type == dqntype.Soft:
        #     disc_act, cont_act = self._soft_train_action(
        #         observations, action_mask, step, debug
        #     )
        #     # print(disc_act)
        #     # print(cont_act)
        #     # print()
        # else:
        disc_act, cont_act = self._e_greedy_train_action(
            observations, action_mask, step, debug
        )
        self.step += int(step)
        return disc_act, cont_act, 0.0, 0.0, 0.0, 0.0

    def ego_actions(self, observations, action_mask=None):
        return 0, 0

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
        values, disc_adv, cont_adv = self.Q1(observations)
        if self.eval_mode:
            return 0, 0
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
                return 0, 0
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
            return dloss, closs

    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action=None, debug=False):
        with torch.no_grad():
            value, dac, cac = self.Q1(obs, legal_action)
            if debug:
                print(f"value: {value}, dac: {dac}, cac: {cac}, eps: {self.eps}")
            if self.dueling:
                return value.mean()  # TODO make sure this doesnt need to be item()

            dq = 0
            n = 0
            if (
                self.discrete_action_dims is not None
                and len(self.discrete_action_dims) > 0
            ):
                n += 1
                for hi, h in enumerate(dac):
                    a = torch.argmax(h, dim=-1)
                    bestq = h[a].item()
                    h[a] = 0
                    if legal_action is not None:
                        if torch.sum(legal_action[hi]) == 1:
                            otherq = (
                                bestq  # no other choices so 100% * only legal choice
                            )
                        else:
                            otherq = torch.sum(  # average of other choices
                                h * legal_action[hi], dim=-1
                            ) / (torch.sum(legal_action[hi], dim=-1) - 1)
                    else:
                        otherq = torch.sum(h, dim=-1) / (
                            self.discrete_action_dims[hi] - 1
                        )
                        if debug:
                            print(
                                f"{otherq} = self.eps * {torch.sum(h, dim=-1)} / ({self.discrete_action_dims[hi] - 1})"
                            )

                    qmean = (1 - self.eps) * bestq + self.eps * otherq
                    if debug:
                        print(
                            f"dq: {qmean} = {(1 - self.eps)} * {bestq} + {self.eps} * {otherq}"
                        )
                    dq += qmean
                dq = dq / len(self.discrete_action_dims)
            cq = 0
            if self.continuous_action_dims > 0:
                n += 1
                for h in cac:
                    a = torch.argmax(h, dim=-1)

                    bestq = h[a].item()
                    h[a] = 0
                    otherq = torch.sum(h, dim=-1) / (self.n_c_action_bins - 1)

                    if debug:
                        print(
                            f"cq: {(1 - self.eps) * bestq + self.eps * otherq} = {(1 - self.eps)} * {bestq} + {self.eps} * ({otherq})"
                        )
                    cq += (1 - self.eps) * bestq + self.eps * otherq
                cq = cq / self.continuous_action_dims

            return value + (cq + dq) / (max(n, 1))

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

    # def old_reinforcement_learn(
    #     self, batch: FlexiBatch, agent_num=0, critic_only=False, debug=False
    # ):
    #     if self.eval_mode:
    #         return 0, 0
    #     if debug:
    #         print("\nDoing Reinforcement learn \n")
    #     dqloss = 0
    #     cqloss = 0
    #     with torch.no_grad():
    #         dQ_ = 0
    #         cQ_ = 0
    #         next_values, next_disc_adv, next_cont_adv = self.Q1(batch.obs_[agent_num])
    #         print(next_cont_adv.shape)
    #         # print(next_values)
    #         dnv_ = 0
    #         cnv_ = 0
    #         if self.dueling:
    #             dnv_ = next_values
    #             cnv_ = (  # Reshaping this to match the shape of next_cont_adv
    #                 next_values.expand(-1, self.continuous_action_dims)
    #                 .unsqueeze(-1)
    #                 .expand(-1, -1, self.n_c_action_bins)
    #             )
    #         if debug:
    #             print(
    #                 f"next vals: {next_values}, next_disct_adv: {next_disc_adv}, next_cont_adv: {next_cont_adv}"
    #             )
    #             # print(f"stacked: {torch.stack(next_cont_adv, dim=1)}")
    #             # input()

    #         if (
    #             self.discrete_action_dims is not None
    #             and len(self.discrete_action_dims) > 0
    #         ):
    #             dQ_ = torch.zeros(
    #                 (batch.global_rewards.shape[0], len(self.discrete_action_dims)),
    #                 dtype=torch.float32,
    #             ).to(self.device)
    #             for i in range(len(self.discrete_action_dims)):
    #                 if self.dqn_type == dqntype.EGreedy:
    #                     dQ_[:, i] = torch.max(
    #                         next_disc_adv[i], dim=-1
    #                     ).values + dnv_.squeeze(-1)
    #                 elif self.dqn_type == dqntype.Soft:
    #                     disc_probs = Categorical(logits=next_disc_adv[i], dim=-1).probs
    #                     dQ_[:, i] = torch.sum(
    #                         disc_probs * (next_disc_adv[i] + dnv_),
    #                         dim=-1,
    #                     )

    #         if (
    #             self.continuous_action_dims is not None
    #             and self.continuous_action_dims > 0
    #         ):
    #             if self.dqn_type == dqntype.EGreedy:
    #                 cQ_ = torch.max(
    #                     (
    #                         torch.stack(next_cont_adv, dim=1)
    #                         + (cnv_ if self.dueling else 0)
    #                     ),
    #                     dim=-1,
    #                 ).values
    #             elif self.dqn_type == dqntype.Soft:
    #                 scq = torch.stack(next_cont_adv, dim=1)
    #                 next_probs = Categorical(logits=scq).probs
    #                 cQ_ = torch.sum(next_probs * (scq + cnv_), dim=-1)

    #     values, disc_adv, cont_adv = self.Q1(batch.obs[agent_num])

    #     discretized_actions = None
    #     discrete_actions = None
    #     dnv = 0
    #     cnv = 0
    #     if self.dueling:
    #         dnv = values.squeeze(-1)
    #         cnv = values.expand(-1, self.continuous_action_dims).unsqueeze(-1)

    #     if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
    #         discrete_actions = batch.discrete_actions[agent_num]
    #         dQ = torch.zeros(
    #             (batch.global_rewards.shape[0], len(self.discrete_action_dims)),
    #             dtype=torch.float32,
    #         ).to(self.device)

    #         for i in range(len(self.discrete_action_dims)):
    #             dQ[:, i] = (
    #                 torch.gather(
    #                     disc_adv[i],
    #                     dim=-1,
    #                     index=batch.discrete_actions[agent_num, :, i].unsqueeze(-1),
    #                 ).squeeze(-1)
    #                 + dnv
    #             )

    #         if self.dqn_type == dqntype.EGreedy:
    #             dqloss = (
    #                 dQ
    #                 - batch.global_rewards.unsqueeze(-1)
    #                 - (self.gamma * (1 - batch.terminated)).unsqueeze(-1) * dQ_
    #             ) ** 2

    #         elif self.dqn_type == dqntype.Soft:
    #             dqloss = (
    #                 dQ
    #                 - batch.global_rewards.unsqueeze(-1)
    #                 - (self.gamma * (1 - batch.terminated)).unsqueeze(-1) * dQ_
    #             ) ** 2
    #             for h in range(len(disc_adv)):
    #                 enloss = (
    #                     Categorical(logits=disc_adv[h]).entropy()
    #                     * self.entropy_loss_coef
    #                 )
    #                 # print(dqloss[:, h].shape, enloss.shape)
    #                 if torch.isnan(enloss).any():
    #                     print("NAN in entropy")
    #                     print(enloss)
    #                 if torch.isnan(dqloss).any():
    #                     print("NAN in dqloss")
    #                     print(dqloss)
    #                 dqloss[:, h] -= enloss
    #         else:
    #             dqloss = 0
    #             for h in range(len(disc_adv)):
    #                 with torch.no_grad():
    #                     next_log_probs = Categorical(logits=next_disc_adv[h]).log_prob(
    #                         torch.argmax(next_disc_adv[h], dim=-1)
    #                     )
    #                     lnprobs = Categorical(logits=disc_adv[h]).log_prob(
    #                         batch.discrete_actions[agent_num, :, h]
    #                     )
    #                     dQ_ = torch.sum(
    #                         next_probs
    #                         * (
    #                             (next_disc_adv[h] + dnv_ if self.dueling else 0)
    #                             - self.entropy_loss_coef * next_log_probs
    #                         ),
    #                         dim=-1,
    #                     )
    #                 dqloss += (
    #                     dQ[:, h]
    #                     - batch.global_rewards
    #                     - (self.munchausen * self.entropy_loss_coef * lnprobs)
    #                     - (self.gamma * (1 - batch.terminated)) * dQ_
    #                 ) ** 2
    #         dqloss = dqloss.mean()

    #     if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
    #         discretized_actions = self._discretize_actions(
    #             batch.continuous_actions[agent_num]
    #         ).unsqueeze(-1)

    #         cQ = (
    #             torch.gather(
    #                 torch.stack(cont_adv, dim=1),
    #                 dim=-1,
    #                 index=discretized_actions,
    #             )
    #             + (cnv if self.dueling else 0)
    #         ).squeeze(-1)

    #         if self.dqn_type == dqntype.EGreedy:
    #             cqloss = (
    #                 cQ
    #                 - (
    #                     batch.global_rewards.unsqueeze(-1)
    #                     + (self.gamma * (1 - batch.terminated)).unsqueeze(-1) * cQ_
    #                 )
    #             ) ** 2

    #         elif self.dqn_type == dqntype.Soft:
    #             cqloss = (
    #                 cQ
    #                 - (
    #                     batch.global_rewards.unsqueeze(-1)
    #                     + (self.gamma * (1 - batch.terminated)).unsqueeze(-1) * cQ_
    #                 )
    #             ) ** 2
    #             cprob = torch.softmax(torch.stack(cont_adv, dim=1), dim=-1)
    #             cqloss -= (
    #                 Categorical(probs=cprob).entropy() * self.entropy_loss_coef
    #             )  # torch.sum(cprob * torch.log(cprob), dim=-1)

    #         else:
    #             cqloss = 0
    #             with torch.no_grad():
    #                 stacknc = torch.stack(next_cont_adv, dim=1)
    #                 next_probs = torch.softmax(stacknc, dim=-1)
    #                 lnprobs = torch.log(
    #                     torch.softmax(torch.stack(cont_adv, dim=1), dim=-1)
    #                     .gather(
    #                         index=discretized_actions,
    #                         dim=-1,
    #                     )
    #                     .squeeze(-1)
    #                 )
    #                 cQ_ = torch.sum(
    #                     next_probs
    #                     * (
    #                         (stacknc + cnv_ if self.dueling else 0)
    #                         - self.entropy_loss_coef * torch.log(next_probs)
    #                     ),
    #                     dim=-1,
    #                 )
    #             cqloss = (
    #                 cQ
    #                 - batch.global_rewards.unsqueeze(-1)
    #                 - (self.munchausen * self.entropy_loss_coef * lnprobs)
    #                 - (self.gamma * (1 - batch.terminated).unsqueeze(-1)) * cQ_
    #             ) ** 2

    #         cqloss = cqloss.mean()

    #     conservative_loss = 0
    #     if self.conservative:
    #         # print(discretized_actions.shape)
    #         conservative_loss = self.cql_loss(
    #             disc_adv,
    #             cont_adv,
    #             discrete_actions,
    #             discretized_actions,
    #         )
    #     loss = dqloss + cqloss + conservative_loss
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     if self.clip_grad is not None and self.clip_grad > 0:
    #         torch.nn.utils.clip_grad_norm_(
    #             self.parameters(),
    #             self.clip_grad,
    #             error_if_nonfinite=True,
    #             foreach=True,
    #         )
    #     self.optimizer.step()

    #     return (
    #         dqloss.item() if torch.is_tensor(dqloss) else dqloss,
    #         cqloss.item() if torch.is_tensor(cqloss) else cqloss,
    #     )  # actor loss, critic loss

    # torch no grad called in reinfrocement learn so no need here
    def _target(
        self,
        values,
        advantages,
        rewards,
        terminated,
        action_dim=None,
        jagged=True,
        debug=True,
    ):
        # print(
        #     f"advantages {advantages.shape if isinstance(advantages, torch.Tensor) else [len(advantages),advantages[0].shape]} jagged: {jagged} values: {values.shape if isinstance(values,torch.Tensor) else 0}"
        # )
        vals = values

        if jagged:  # discrete action bins are jagget
            assert (
                action_dim is not None
            ), "Cant be jagged=True with no action dim passed to _target()"
            # vals = values.squeeze(-1) if self.dueling else 0  # make it a column vector
            # action_dim = len(self.discrete_action_dims)
            Q_ = torch.zeros(
                size=(advantages[0].shape[0], len(action_dim)),
                device=self.device,
                dtype=torch.float32,
            )
            if debug:
                print("Jagged target() q shape, adv shape, and value shape")
                print(Q_.shape, vals)
                for i in range(len(action_dim)):
                    print("  " + str(advantages[i].shape))
                # print(advantages)

            for i in range(len(action_dim)):
                # Treat actions as probabalistic if using soft Q or m-dqn
                if self.dqn_type == dqntype.Munchausen or self.dqn_type == dqntype.Soft:
                    lprobs = torch.log_softmax(
                        advantages[i] / self.entropy_loss_coef, dim=-1
                    )
                    probs = torch.exp(lprobs)
                    if debug:
                        print(
                            f"  M-DQN or Soft DQN: adv(max_a): {torch.max(advantages[i], dim=-1).values.shape}, vals: {vals.shape if self.dueling and isinstance(vals, torch.Tensor) else values}"
                        )

                    Q_[:, i] = torch.sum(
                        probs * (advantages[i] - self.entropy_loss_coef * lprobs),
                        dim=-1,
                    )
                else:
                    if debug:
                        print(
                            f"  Standard DQN: adv(max_a): {torch.max(advantages[:,i], dim=-1).values.shape}, vals: {vals.shape if self.dueling and isinstance(vals, torch.Tensor) else values}"
                        )
                    Q_[:, i] = torch.max(advantages[i], dim=-1).values
            Q_ = Q_.sum(dim=-1) + (
                vals.squeeze(-1) if isinstance(vals, torch.Tensor) else 0
            )

        else:  # continuous bins are not jagged
            if debug:
                print(f"  Not Jagget target advantages: {advantages.shape}")
            # advantages = advantages.transpose(0, 1)
            # Treat actions as probabalistic if using soft Q or m-dqn
            if self.dqn_type == dqntype.Munchausen or self.dqn_type == dqntype.Soft:
                lprobs = torch.log_softmax(advantages / self.entropy_loss_coef, dim=-1)
                probs = torch.exp(lprobs)
                if debug:
                    print(
                        f"  M-DQN or Soft DQN: adv(max_a): {torch.max(advantages, dim=-1).values.shape}, vals: {values.shape if self.dueling else values}"
                    )

                # if isinstance(vals, torch.Tensor):
                #     vals = vals.unsqueeze(-1)
                #     if len(vals.shape) != len(advantages.shape):
                #         print(
                #             f"max advs: {advantages.shape} + vals {vals.shape if isinstance(vals,torch.Tensor) else 0}"
                #         )
                #         input("hmm this looks bad: ")

                # q_vals = vals + advantages
                # print(f"vals before q soft sum not jagged: {vals.shape}")
                Q_ = (
                    torch.sum(
                        probs * (advantages - self.entropy_loss_coef * lprobs), dim=-1
                    )
                ).sum(dim=-1) + (
                    vals.squeeze(-1) if isinstance(vals, torch.Tensor) else 0.0
                )

                if debug:
                    print(
                        f"  vals shape: {vals.shape if self.dueling and isinstance(vals, torch.Tensor) else 0} Q_: {Q_.shape}, rewards: {rewards.unsqueeze(-1).shape}, terminated: {terminated.unsqueeze(-1).shape}"
                    )
            else:
                if self.dueling:
                    vals = values
                else:
                    vals = 0
                if debug:
                    print(
                        f"  Standard DQN: adv(max_a): {torch.max(advantages, dim=-1).values.shape}, vals: {vals.shape if self.dueling and isinstance(vals, torch.Tensor) else values}"
                    )
                # print(
                #     f"max advs: {torch.max(advantages, dim=-1).values.shape} + vals {vals.shape if isinstance(vals,torch.Tensor) else 0}"
                # )
                # if isinstance(vals, torch.Tensor):
                #     if len(vals.shape) != len(
                #         torch.max(advantages, dim=-1).values.shape
                #     ):
                #         print(
                #             f"max advs: {advantages.shape} + vals {vals.shape if isinstance(vals,torch.Tensor) else 0}"
                #         )
                #         input("uh oh: ")
                # print(
                #     f"advmax: {torch.max(advantages, dim=-1).values.shape} + vals {vals.shape if isinstance(vals,torch.Tensor) else 0.0}"
                # )
                Q_ = torch.max(advantages, dim=-1).values.sum(dim=-1) + (
                    vals.squeeze(-1) if isinstance(vals, torch.Tensor) else 0.0
                )

        if debug:
            print(
                f"  Q_: {Q_.shape}, rewards: {rewards.shape}, terminated: {terminated.shape}"
            )
            input()

        targets = rewards + (self.gamma * (1 - terminated)) * Q_
        if debug:
            print(f"  targets: {targets.shape}")
        return targets

    def reinforcement_learn(
        self, batch: FlexiBatch, agent_num=0, critic_only=False, debug=False
    ):
        self.update_num += 1
        if self.eval_mode:
            return float(0.0), float(0.0)

        dqloss, cqloss = 0, 0
        continuous_actions = None
        discrete_actions = None
        if self.discrete_action_dims is not None:
            discrete_actions = batch.discrete_actions[agent_num]  # type: ignore
        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            continuous_actions = self._discretize_actions(
                batch.continuous_actions[agent_num]  # type: ignore
            )
            # print("Discretized batch historical actions")
            # print(continuous_actions.shape)
            # input("Does this make sense")
            # print(batch.continuous_actions[agent_num])
            # print()
        if debug:
            if continuous_actions is not None:
                print(f"Continuous actions: {continuous_actions.shape}")
            if discrete_actions is not None:
                print(f"Discrete actions: {discrete_actions.shape}")
            print(
                f"Batch obs: {batch.obs[agent_num].shape}, Batch obs_: {batch.obs_[agent_num].shape}"
            )
        discrete_target = 0
        continuous_target = 0
        values, disc_adv, cont_adv = self.Q1(batch.obs[agent_num])
        # print(batch.obs[agent_num])
        # if cont_adv is not None:
        #    cont_adv = cont_adv.transpose(0, 1)
        with torch.no_grad():
            next_values, next_disc_adv, next_cont_adv = self.Q2(batch.obs_[agent_num])

            # print(
            #     f" in rl learn: next_vals: {next_values.shape if isinstance(next_values, torch.Tensor) else 0} nextdadv: {next_disc_adv} nextcadv: {next_cont_adv}"
            # )

            if (
                self.discrete_action_dims is not None
                and len(self.discrete_action_dims) > 0
            ):
                assert (
                    discrete_actions is not None
                ), "Cant learn on discrete actions if they are None"
                if debug:
                    print("Testing discrete Targets")
                discrete_target = self._target(
                    values=next_values,
                    advantages=next_disc_adv,
                    rewards=batch.global_rewards,
                    terminated=batch.terminated,
                    action_dim=self.discrete_action_dims,
                    jagged=True,
                    debug=debug,
                )
                if self.dqn_type == dqntype.Munchausen:
                    with torch.no_grad():
                        for i in range(len(self.discrete_action_dims)):
                            # temp_disc_adv = disc_adv[i].detach()
                            # if munchausen add tau*alpha*lp(a|s) to target
                            print(
                                f"disc_adv shape {len(disc_adv)} {disc_adv[0].shape} {discrete_actions.shape}"
                            )
                            print(
                                f"dtarg: {discrete_target.shape} dacts: {discrete_actions[:,i].shape} lp shape: {Categorical( logits=disc_adv[i] / self.entropy_loss_coef).log_prob(discrete_actions[:,i]).shape}"
                            )
                            discrete_target += (
                                self.entropy_loss_coef
                                * self.munchausen
                                * (
                                    Categorical(
                                        logits=disc_adv[i] / self.entropy_loss_coef
                                    ).log_prob(discrete_actions[:, i])
                                )
                            )
            if (
                self.continuous_action_dims is not None
                and self.continuous_action_dims > 0
            ):
                assert (
                    continuous_actions is not None
                ), "Cant learn on continuous actions if they are None"
                if debug:
                    print("Testing continuous Targets")
                # if cont_adv is not None:
                #    cont_adv = cont_adv.transpose(0, 1)
                continuous_target = self._target(
                    values=next_values,
                    advantages=next_cont_adv,
                    rewards=batch.global_rewards,
                    terminated=batch.terminated,
                    jagged=False,
                    debug=debug,
                )
                # cont_adv = cont_adv.transpose(0, 1)
                if self.dqn_type == dqntype.Munchausen:
                    with torch.no_grad():
                        # THESE DO THE EXACT SAME THING
                        # temp_cont_adv = cont_adv.detach()
                        # print(f"cont adv shape: {cont_adv.shape}")
                        # print(f"cact shape: {continuous_actions.shape}")
                        # catprobs = Categorical(
                        #     logits=cont_adv / self.entropy_loss_coef
                        # ).log_prob(continuous_actions)
                        # otherprobs = (
                        #     torch.log_softmax(cont_adv / self.entropy_loss_coef, dim=-1)
                        #     .gather(dim=-1, index=continuous_actions.unsqueeze(-1))
                        #     .squeeze(-1)
                        # )
                        print(
                            f"ct: {continuous_target.shape} log_soft shape: {torch.log_softmax(cont_adv / self.entropy_loss_coef, dim=-1).shape} gather sum shape: {torch.log_softmax(cont_adv / self.entropy_loss_coef, dim=-1).gather(dim=-1, index=continuous_actions.unsqueeze(-1)).squeeze(-1).sum(-1).shape}"
                        )
                        continuous_target = (
                            continuous_target
                            + self.entropy_loss_coef
                            * self.munchausen
                            * torch.log_softmax(
                                cont_adv / self.entropy_loss_coef, dim=-1
                            )
                            .gather(dim=-1, index=continuous_actions.unsqueeze(-1))
                            .squeeze(-1)
                            .sum(-1)
                        )
        cQ = torch.zeros(1, device=self.device)
        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            assert (
                continuous_actions is not None
            ), "Cant do continuous action update if actions are None"
            assert isinstance(
                continuous_target, torch.Tensor
            ), "Need a target tensor if continuous action dims are to be used"
            if debug:
                print(
                    f"Calculating cQ: cont_advs.shape: {cont_adv.shape}, continuous_actions: {continuous_actions.unsqueeze(-1).shape}, vals: {values.shape if self.dueling else 0}"
                )
            # print(
            #     f"cont adv: {cont_adv.shape} cactions: {continuous_actions.unsqueeze(-1).shape}"
            # )
            # input(
            #     f"cQ; {torch.gather(input=cont_adv,dim=-1,index=continuous_actions.unsqueeze(-1),).shape} vals: {values.shape if isinstance(values,torch.Tensor) else 0.0}"
            # )
            cQ = torch.gather(
                input=cont_adv,
                dim=-1,
                index=continuous_actions.unsqueeze(-1),
            ).squeeze(-1).sum(-1) + (values.squeeze(-1) if self.dueling else 0)

            if debug:
                print(f"cQ: {cQ.shape}, continuous_target: {continuous_target.shape}")

        dQ = torch.zeros(1, device=self.device)
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            assert (
                discrete_actions is not None
            ), "Cant do discrete update if discrete actions are None"
            dQ = torch.zeros(
                size=(batch.global_rewards.shape[0], len(self.discrete_action_dims)),
                device=self.device,
                dtype=torch.float32,
            )
            for d in range(len(self.discrete_action_dims)):
                if debug:
                    print(
                        f"  disc_adv: {disc_adv[d].shape}, disc_act: {discrete_actions[:, d].unsqueeze(-1).shape}"
                    )
                # input(
                #     f"dQ[{d}]; {torch.gather(disc_adv[d],dim=-1,index=discrete_actions[:, d].unsqueeze(-1),).shape} vals: {values.shape if isinstance(values,torch.Tensor) else 0.0}"
                # )
                dQ[:, d] = torch.gather(
                    disc_adv[d],
                    dim=-1,
                    index=discrete_actions[:, d].unsqueeze(-1),
                ).squeeze(-1)

            # print(
            #     f"dq before sum: {dQ.shape} v: {values.squeeze(-1).shape if isinstance(values,torch.Tensor) else 0.0}"
            # )
            dQ = dQ.sum(-1) + (
                values.squeeze(-1) if isinstance(values, torch.Tensor) else 0.0
            )
            # print(f"new dq: {dQ.shape}")
        dqloss, cqloss = 0, 0
        trainable = False
        if self.discrete_action_dims is not None and len(self.discrete_action_dims) > 0:
            # print(dQ.shape)
            # print(discrete_target.shape)
            dqloss = (dQ - discrete_target) ** 2
            dqloss = dqloss.mean()
            trainable = True

        if self.continuous_action_dims is not None and self.continuous_action_dims > 0:
            # print(cQ.shape)
            # print(f"ctarg: {continuous_target.shape}")
            # print(f"cQ: {cQ.shape}")
            cqloss = (cQ - continuous_target) ** 2
            cqloss = cqloss.mean()
            trainable = True
            # input("mm")

        if trainable:
            loss = dqloss + cqloss
            # print(loss)
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
            self.optimizer.step()
        else:
            warnings.warn(
                "Action dims both zero so there is nothing to train. Not updating the model."
            )
        if dqloss != 0:
            dqloss = dqloss.item()
        if cqloss != 0:
            cqloss = cqloss.item()

        if self.update_num > 1:
            # print("updating delayed network")
            self.update_num = 0
            with torch.no_grad():
                self.Q2.load_state_dict(self.Q1.state_dict())

        return float(dqloss), float(cqloss)  # actor loss, critic loss

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
                value_per_head=False,
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
                value_per_head=False,
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
