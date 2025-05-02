from .Agent import ValueS, MixedActor, Agent
from .Util import T
import torch
from flexibuff import FlexiBatch
from torch.distributions import Categorical
import numpy as np
import torch.nn as nn
import pickle
import os


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
        starting_actorlogstd=0,
        clip_grad=True,
        gae_lambda=0.95,
        load_from_checkpoint=None,
        name="PPO",
        eval_mode=False,
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
            "starting_actorlogstd",
            "clip_grad",
            "gae_lambda",
            "g_mean",
            "steps",
            "eval_mode",
        ]
        assert (
            continuous_action_dim > 0 or discrete_action_dims is not None
        ), "At least one action dim should be provided"
        self.name = name
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

        self.starting_actorlogstd = starting_actorlogstd
        self.g_mean = 0
        self.steps = 0
        self.anneal_lr = anneal_lr
        self.lr = lr

        self._get_torch_params(starting_actorlogstd)

        if self.continuous_action_dim is not None and self.continuous_action_dim > 0:
            self.min_actions = torch.from_numpy(min_actions).to(self.device)
            self.max_actions = torch.from_numpy(max_actions).to(self.device)

    def _get_torch_params(self, starting_actorlogstd):
        self.actor = MixedActor(
            obs_dim=self.obs_dim,
            continuous_action_dim=self.continuous_action_dim,
            discrete_action_dims=self.discrete_action_dims,
            max_actions=self.max_actions,
            min_actions=self.min_actions,
            hidden_dims=self.hidden_dims,
            device=self.device,
            orthogonal_init=self.orthogonal,
            activation=self.activation,
        )

        self.critic = ValueS(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dims[0],
            device=self.device,
            orthogonal_init=self.orthogonal,
            activation=self.activation,
        )
        self.actor_logstd = (
            nn.Parameter(
                torch.zeros(1, self.continuous_action_dim), requires_grad=True
            ).to(self.device)
            + starting_actorlogstd
        )
        # print(self.actor_logstd)
        self.actor_logstd.retain_grad()

        self.optimizer = torch.optim.Adam(
            list(self.parameters()) + [self.actor_logstd], lr=self.lr
        )

    def _sample_multi_discrete(
        self, logits, debug=False
    ):  # logits of the form [action_dim, batch_size, action_dim_size]
        actions = torch.zeros(
            size=(len(self.discrete_action_dims),),
            device=self.device,
            dtype=torch.int,
        )
        log_probs = torch.zeros(
            size=(len(self.discrete_action_dims),),
            device=self.device,
        )
        for i in range(len(self.discrete_action_dims)):
            # print(f"logits: {logits}")
            dist = Categorical(probs=logits[i])
            actions[i] = dist.sample()
            # print(f"act: {actions[i]}")
            # print(
            #    f"logprob: {dist.log_prob(actions[i])}, {torch.log(logits[i][actions[i]])}"
            # )
            log_probs[i] = dist.log_prob(actions[i])
            # print(dist)
        return actions, log_probs

    def train_actions(self, observations, action_mask=None, step=False, debug=False):
        if debug:
            print(f"  Testing Train Actions: Observations: {observations}")
        if not torch.is_tensor(observations):
            observations = T(observations, device=self.device, dtype=torch.float)
        if not torch.is_tensor(action_mask) and action_mask is not None:
            action_mask = torch.tensor(action_mask, dtype=torch.float).to(self.device)

        if debug:
            print(f"  After tensor check: Observations{observations}")
        # print(f"Observations: {observations.shape} {observations}")

        if step:
            self.steps += 1
        if self.anneal_lr > 0:
            frac = max(1.0 - (self.steps - 1.0) / self.anneal_lr, 0.001)
            lrnow = frac * self.lr
            self.optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            continuous_logits, discrete_logits = self.actor(
                x=observations, action_mask=action_mask, gumbel=False, debug=False
            )
            if debug:
                print(f"  After actor: clog {continuous_logits}, dlog{discrete_logits}")

        if debug:
            print(
                f" Expanding actor logstd {self.actor_logstd.squeeze(0)}, {continuous_logits}"
            )
        continuous_actions, continuous_log_probs = None, None
        try:
            if self.continuous_action_dim > 0:
                # action_logstd = self.actor_logstd
                action_std = torch.exp(self.actor_logstd.squeeze(0))
                continuous_dist = torch.distributions.Normal(
                    loc=continuous_logits,
                    scale=action_std,
                )
                continuous_actions = continuous_dist.sample()
                continuous_log_probs = (
                    continuous_dist.log_prob(continuous_actions).detach().cpu().numpy()
                )
                continuous_actions = continuous_actions.detach().cpu().numpy()
        except Exception as e:
            print(
                f"bad stuff, {continuous_logits}, {discrete_logits}, {observations}, {action_mask} {e}"
            )
            exit()

        discrete_actions, discrete_log_probs = None, None
        if self.discrete_action_dims is not None:
            # print(f"obs: {observations}")

            discrete_actions, discrete_log_probs = self._sample_multi_discrete(
                discrete_logits
            )
            # print(
            #     "Logit, act, logprob",
            #     discrete_logits,
            #     discrete_actions,
            #     discrete_log_probs,
            # )
            # print(torch.log(discrete_logits[0][discrete_actions[0]]))
            discrete_actions = discrete_actions.detach().cpu().numpy()
            discrete_log_probs = discrete_log_probs.detach().cpu().numpy()

        return (
            discrete_actions,
            continuous_actions,
            discrete_log_probs,
            continuous_log_probs,
            0,  # vals.detach().cpu().numpy(), TODO: re-enable this when flexibuff is done
        )

    # takes the observations and returns the action with the highest probability
    def ego_actions(self, observations, action_mask=None):
        with torch.no_grad():
            continuous_actions, discrete_action_activations = self.actor(
                observations, action_mask, gumbel=False
            )
            if len(continuous_actions.shape) == 1:
                continuous_actions = continuous_actions.unsqueeze(0)
            # Ignore the continuous actions std for ego action
            discrete_actions = torch.zeros(
                (observations.shape[0], len(discrete_action_activations)),
                device=self.device,
                dtype=torch.float32,
            )
            for i, activation in enumerate(discrete_action_activations):
                discrete_actions[:, i] = torch.argmax(activation, dim=1)
            return discrete_actions, continuous_actions

    def imitation_learn(
        self,
        observations,
        continuous_actions=None,
        discrete_actions=None,
        action_mask=None,
    ):

        dact, cact = self.actor(
            observations, action_mask=action_mask, gumbel=False, debug=False
        )
        loss = 0
        if self.continuous_action_dim > 0:
            loss += ((cact - continuous_actions) ** 2).mean()
        if self.discrete_action_dims is not None:
            for head in range(len(self.discrete_action_dims)):
                loss += nn.CrossEntropyLoss()(
                    dact[head], discrete_actions[:, head].long()
                )
        loss = torch.nn.functional.cross_entropy(probs, oh_actions, reduction="mean")
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()  # loss

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

    def marl_learn(self, batch, agent_num, mixer, critic_only=False, debug=False):
        return super().marl_learn(batch, agent_num, mixer, critic_only, debug)

    def zero_grads(self):
        return 0

    def _get_disc_log_probs_entropy(self, logits, actions):
        log_probs = torch.zeros_like(actions, dtype=torch.float)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        return log_probs, dist.entropy().mean()

    def _get_cont_log_probs_entropy(self, logits, actions):
        log_probs = torch.zeros_like(actions, dtype=torch.float)
        dist = torch.distributions.Normal(
            loc=logits, scale=torch.exp(self.actor_logstd.expand_as(logits))
        )
        log_probs = dist.log_prob(actions)
        return log_probs, dist.entropy().mean()

    def _get_probs_and_entropy(self, batch: FlexiBatch, agent_num):
        cp, dp = self.actor(
            batch.obs[agent_num], action_mask=batch.action_mask[agent_num]
        )
        if len(self.discrete_action_dims) > 0:
            old_disc_log_probs = []
            old_disc_entropy = []
            for head in range(len(self.discrete_action_dims)):
                odlp, ode = self._get_disc_log_probs_entropy(
                    logits=dp[head],
                    actions=batch.discrete_actions[agent_num][:, head],
                )
                old_disc_log_probs.append(odlp)
                old_disc_entropy.append(ode)
        else:
            old_disc_log_probs = 0
            old_disc_entropy = 0

        if self.continuous_action_dim > 0:
            old_cont_log_probs, old_cont_entropy = self._get_cont_log_probs_entropy(
                logits=cp, actions=batch.continuous_actions[agent_num]
            )
        else:
            old_cont_log_probs = 0
            old_cont_entropy = 0

        return (
            old_disc_log_probs,
            old_disc_entropy,
            old_cont_log_probs,
            old_cont_entropy,
        )

    def _G(self, batch, agent_num):
        G = torch.zeros_like(batch.global_rewards).to(self.device)
        G[-1] = batch.global_rewards[-1]
        if batch.terminated[-1] < 0.5:
            if self.advantage_type == "constant":
                G[-1] += self.gamma * self.g_mean
            else:
                G[-1] += self.gamma * self.critic(batch.obs_[agent_num][-1]).squeeze(-1)

        for i in range(len(batch.global_rewards) - 2, -1, -1):
            G[i] = batch.global_rewards[i] + self.gamma * G[i + 1] * (
                1 - batch.terminated[i]
            )
        G = G.unsqueeze(-1)
        return G

    def _gae(self, batch, agent_num):
        with torch.no_grad():
            advantages = torch.zeros_like(batch.global_rewards).to(self.device)
            num_steps = batch.global_rewards.shape[0]
            last_values = self.critic(batch.obs_[agent_num, -1]).squeeze(-1)
            values = self.critic(batch.obs[agent_num]).squeeze(-1)

            last_gae_lam = 0
            for step in reversed(range(num_steps)):
                if step == num_steps - 1:
                    next_non_terminal = 1.0 - batch.terminated[-1]
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - batch.terminated[step]
                    next_values = values[step + 1]
                delta = (
                    batch.global_rewards[step]
                    + self.gamma * next_values * next_non_terminal
                    - values[step]
                )
                last_gae_lam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                )
                advantages[step] = last_gae_lam
            # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
            # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
            #
            G = advantages + values

            # print(f"new: advantages {advantages} and G {G}")
            # input()
        return G.unsqueeze(-1), advantages.unsqueeze(-1)

    def _td(self, batch, agent_num):
        reward_arr = batch.global_rewards
        td = torch.zeros_like(reward_arr).to(self.device)

        with torch.no_grad():  # If last obs is non terminal critic to not bias it
            old_values = self.critic(batch.obs[agent_num]).squeeze(-1)
            td[-1] = (
                self.gamma
                * self.critic(batch.obs_[agent_num, -1]).squeeze(-1)
                * batch.terminated[-1]
                - old_values[-1]
            )

        for t in range(len(reward_arr) - 1):
            td[t] = (
                reward_arr[t]
                + self.gamma * old_values[t + 1] * (1 - batch.terminated[t])
                - old_values[t]
            )

        G = td + old_values
        return G.unsqueeze(-1), td.unsqueeze(-1)

    def _print_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p is None or p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        print(total_norm)

    def reinforcement_learn(
        self,
        batch: FlexiBatch,
        agent_num=0,
        critic_only=False,
        debug=False,
        conenv=False,
    ):
        if self.eval_mode:
            return 0, 0
        # print(f"Doing PPO learn for agent {agent_num}")
        # Update the critic with Bellman Equation
        # Monte Carlo Estimate of returns
        if debug:
            print(f"Starting Reinforcement Learn for agent {agent_num}")
        # # G = G / 100
        with torch.no_grad():
            if self.advantage_type == "gv":
                G = self._G(batch, agent_num)
                advantages = G - self.critic(batch.obs[agent_num])
            elif self.advantage_type == "gae":
                G, advantages = self._gae(batch, agent_num)
            elif self.advantage_type == "a2c":
                G, advantages = self._td(batch, agent_num)
            elif self.advantage_type == "constant":
                G = self._G(batch, agent_num)
                self.g_mean = 0.9 * self.g_mean + 0.1 * G.mean()
                advantages = G - self.g_mean
            elif self.advantage_type == "g":
                G = self._G(batch, agent_num)
                advantages = G

            else:
                raise ValueError("Invalid advantage type")

                # print(advantages.squeeze(-1))
            if debug:
                print(f"  batch rewards: {batch.global_rewards}")
                print(f"  raw critic: {self.critic(batch.obs[agent_num])}")
                print(f"  Advantages: {advantages}")
                print(f"  G: {G}")
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        avg_actor_loss = 0
        avg_critic_loss = 0
        # Update the actor
        action_mask = None
        if batch.action_mask is not None:
            action_mask = batch.action_mask[agent_num]  # TODO: Unit test this later

        bsize = len(batch.global_rewards)
        # print(bsize)
        # print(self.mini_batch_size)
        nbatch = bsize // self.mini_batch_size
        mini_batch_indices = np.arange(len(batch.global_rewards))
        # np.random.shuffle(mini_batch_indices)

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
                # print(indices)
                # print(bstart, bend)
                # input()
                bnum += 1
                if debug:
                    print(
                        f"    Mini batch: {bstart}:{bend}, Indices: {indices}, {len(indices)}"
                    )

                V_current = self.critic(batch.obs[agent_num, indices])
                if debug:
                    print(
                        f"    V_current: {V_current.shape}, G[indices] {G[indices].shape}"
                    )
                    input()
                # print(V_current)
                # print(G[indices])
                # print(V_current - G[indices])
                # input()
                critic_loss = 0.5 * ((V_current - G[indices]) ** 2).mean()
                # print(torch.abs(V_current - G[indices]).mean())
                if not critic_only:
                    mb_adv = advantages[indices]

                    actor_loss = 0
                    cont_probs, disc_probs = self.actor(
                        batch.obs[agent_num, indices],
                        action_mask=action_mask,  # TODO fix action mask by indices
                        gumbel=False,
                    )
                    if self.continuous_action_dim > 0:
                        if debug:
                            print(f"    cont probs: {cont_probs.shape}")
                            print(
                                f"    logstd: {self.actor_logstd.expand_as(cont_probs)}"
                            )
                            print(f"    advantages: {mb_adv[indices]}")
                        # input("what is up with continuous probabilities")
                        continuous_dist = torch.distributions.Normal(
                            loc=cont_probs,
                            scale=torch.exp(self.actor_logstd.expand_as(cont_probs)),
                        )
                        # print(batch.continuous_actions[agent_num, indices])
                        continuous_log_probs = continuous_dist.log_prob(
                            batch.continuous_actions[agent_num, indices]
                        )

                        if self.ppo_clip > 0:
                            logratio = (
                                continuous_log_probs
                                - batch.continuous_log_probs[agent_num, indices]
                            )

                            ratio = logratio.exp()

                            pg_loss1 = mb_adv * ratio
                            pg_loss2 = mb_adv * torch.clamp(
                                ratio, 1 - self.ppo_clip, 1 + self.ppo_clip
                            )
                            continuous_policy_gradient = torch.min(pg_loss1, pg_loss2)

                        else:
                            continuous_policy_gradient = continuous_log_probs * mb_adv

                        if debug:
                            print(f"    continuous_log_probs: {continuous_log_probs}")
                            print(
                                f"    continuous_policy_gradient: {continuous_policy_gradient}"
                            )
                        # print(continuous_policy_gradient)
                        actor_loss += (
                            -self.policy_loss * continuous_policy_gradient.mean()
                            - self.entropy_loss * continuous_dist.entropy().mean()
                        )
                        # print(self.actor_logstd.exp())
                        # print(continuous_dist.loc.mean())

                    if self.discrete_action_dims is not None:
                        for head in range(len(self.discrete_action_dims)):
                            if debug:
                                print(f"    Discrete head: {head}")
                                print(f"    disc_probs: {disc_probs[head]}")
                                print(
                                    f"    batch.discrete_actions: {batch.discrete_actions[agent_num,indices,head]}"
                                )
                            probs: torch.Tensor = disc_probs[head]  # Categorical()
                            dist = Categorical(probs=probs)
                            entropy = dist.entropy().mean()
                            # print(probs)
                            # print(batch.discrete_actions[agent_num, indices, head])
                            # selectedprobs = probs.gather(
                            #     -1,
                            #     batch.discrete_actions[
                            #         agent_num, indices, head
                            #     ].unsqueeze(-1),
                            # ).squeeze(-1)
                            # print(selectedprobs)

                            selected_log_probs = dist.log_prob(
                                batch.discrete_actions[agent_num, indices, head]
                            )

                            if self.ppo_clip > 0:

                                logratio = (
                                    selected_log_probs
                                    - batch.discrete_log_probs[agent_num, indices, head]
                                )
                                ratio = logratio.exp()
                                pg_loss1 = mb_adv.squeeze(-1) * ratio
                                pg_loss2 = mb_adv.squeeze(-1) * torch.clamp(
                                    ratio, 1 - self.ppo_clip, 1 + self.ppo_clip
                                )
                                discrete_policy_gradient = torch.min(pg_loss1, pg_loss2)
                            else:
                                discrete_policy_gradient = (
                                    selected_log_probs * mb_adv.squeeze(-1)
                                )

                            actor_loss += (
                                -self.policy_loss * discrete_policy_gradient.mean()
                                - self.entropy_loss * entropy
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

                    avg_actor_loss += actor_loss.item()
                    avg_critic_loss += critic_loss.item()
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


if __name__ == "__main__":
    obs_dim = 3
    continuous_action_dim = 2
    agent = PG(
        obs_dim=obs_dim,
        continuous_action_dim=continuous_action_dim,
        max_actions=np.array([1, 2]),
        min_actions=np.array([0, 0]),
        discrete_action_dims=[4, 5],
        hidden_dims=[32, 32],
        device="cuda:0",
        lr=0.001,
        activation="relu",
        advantage_type="G",
        norm_advantages=True,
        mini_batch_size=7,
        n_epochs=2,
    )
    obs = np.random.rand(obs_dim).astype(np.float32)
    obs_ = np.random.rand(obs_dim).astype(np.float32)
    obs_batch = np.random.rand(14, obs_dim).astype(np.float32)
    obs_batch_ = obs_batch + 0.1

    dacs = np.stack(
        (np.random.randint(0, 4, size=(14)), np.random.randint(0, 5, size=(14))),
        axis=-1,
    )

    mem = FlexiBatch(
        obs=np.array([obs_batch]),
        obs_=np.array([obs_batch_]),
        continuous_actions=np.array([np.random.rand(14, 2).astype(np.float32)]),
        discrete_actions=np.array([dacs]),
        global_rewards=np.random.rand(14).astype(np.float32),
        terminated=np.random.randint(0, 2, size=14),
    )
    mem.to_torch("cuda:0")

    d_acts, c_acts, d_log, c_log, _ = agent.train_actions(obs, step=True, debug=True)
    print(f"Training actions: c: {c_acts}, d: {d_acts}, d_log: {d_log}, c_log: {c_log}")

    for adv_type in ["g", "gae", "a2c", "constant", "gv"]:
        agent.advantage_type = adv_type
        print(f"Reinforcement learning with advantage type {adv_type}")
        aloss, closs = agent.reinforcement_learn(mem, 0, critic_only=False, debug=True)
        print("Done")
        input("Check next one?")

    print("Finished Testing")
