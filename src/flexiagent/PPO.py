from Agent import ValueS, MixedActor, Agent
import torch
from flexibuff import FlexiBatch
from torch.distributions import Categorical


class PPO(Agent):
    def __init__(
        self,
        obs_dim,
        continuous_action_dim=0,
        max_actions=None,
        min_actions=None,
        discrete_action_dims=None,
        lr_actor=0.001,
        lr_critic=0.003,
        gamma=0.99,
        eps_clip=0.2,
        n_epochs=5,
        device="cpu",
        entropy_loss=0.05,
        encoder_dims=[256, 256],
    ):
        super().__init__()
        assert (
            continuous_action_dim > 0 or discrete_action_dims is not None
        ), "At least one action dim should be provided"
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.obs_dim = obs_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = discrete_action_dims
        self.n_epochs = n_epochs

        self.policy_loss = 1
        self.critic_loss = 1
        self.entropy_loss = entropy_loss

        self.actor = MixedActor(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            hidden_dims=encoder_dims,
            device=device,
        )
        self.actor_old = MixedActor(
            obs_dim=obs_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_dims=discrete_action_dims,
            max_actions=max_actions,
            min_actions=min_actions,
            hidden_dims=encoder_dims,
            device=device,
        )
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = ValueS(state_size=obs_dim, hidden_size=256, device=self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def _sample_multi_discrete(
        self, logits, debug=False
    ):  # logits of the form [action_dim, batch_size, action_dim_size]
        actions = torch.zeros(
            size=(logits[0].shape[-1], len(self.discrete_action_dims))
        )
        log_probs = torch.zeros(
            size=(logits[0].shape[-1], len(self.discrete_action_dims))
        )
        for i in range(len(self.discrete_action_dims)):
            dist = Categorical(logits=logits[i])
            actions[:, i] = dist.sample()
            log_probs[:, i] = dist.log_prob(actions[i])
        return actions, log_probs

    def train_action(self, observations, action_mask=None, step=False):
        if not torch.is_tensor(observations):
            observations = torch.tensor(observations, dtype=torch.float).to(self.device)
        if not torch.is_tensor(action_mask) and action_mask is not None:
            action_mask = torch.tensor(action_mask, dtype=torch.float).to(self.device)
        continuous_logits, descrete_logits = self.actor(
            x=observations, action_mask=action_mask, gumbel=False, debug=False
        )
        if len(continuous_logits.shape) == 1:
            continuous_logits = continuous_logits.unsqueeze(0)

        continuous_dist = torch.distributions.Normal(
            loc=continuous_logits[:, : self.continuous_action_dim],
            scale=torch.exp(continuous_logits[:, self.continuous_action_dim :]),
        )
        discrete_actions, discrete_log_probs = self._sample_multi_discrete(
            descrete_logits
        )
        continuous_actions = continuous_dist.sample()
        continuous_log_probs = continuous_dist.log_prob(continuous_actions)
        vals = self.critic(observations).detach()
        return (
            discrete_actions,
            continuous_actions,
            discrete_log_probs,
            continuous_log_probs,
            vals,
        )

    # takes the observations and returns the action with the highest probability
    def ego_action(self, observations, action_mask=None):
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
            return discrete_actions, continuous_actions[:, : self.continuous_action_dim]

    def imitation_learn(self, observations, actions, action_mask=None):
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.int).to(self.device)
        if not torch.is_tensor(observations):
            observations = torch.tensor(observations, dtype=torch.float).to(self.device)

        act, probs, log_probs = self.actor.evaluate(
            observations, action_mask=action_mask
        )
        # max_actions = act.argmax(dim=-1, keepdim=True)
        # loss is MSE loss beteen the actions and the predicted actions
        oh_actions = torch.nn.functional.one_hot(
            actions.squeeze(-1), self.actor_size
        ).float()
        # print(oh_actions.shape, probs.shape)
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

    def expected_V(self, obs, legal_action):
        return self.critic(obs)

    def marl_learn(self, batch, agent_num, mixer, critic_only=False, debug=False):
        return super().marl_learn(batch, agent_num, mixer, critic_only, debug)

    def zero_grads(self):
        return 0

    def reinforcement_learn(
        self, batch: FlexiBatch, agent_num=0, critic_only=False, debug=False
    ):
        print(f"Doing PPO learn for agent {agent_num}")
        # Update the critic with Bellman Equation

        # Monte Carlo Estimate of returns
        G = torch.zeros_like(batch.global_rewards).to(self.device)
        G[-1] = batch.global_rewards[-1]
        for i in range(len(batch.global_rewards) - 2, 0, -1):
            G[i] = batch.global_rewards[i] + self.gamma * G[i + 1] * (
                1 - batch.terminated[i]
            )
        G = G.unsqueeze(-1)
        with torch.no_grad():
            advantages = G - self.critic(batch.obs)

        avg_actor_loss = 0
        # Update the actor
        action_mask = None
        if batch.action_mask is not None:
            action_mask = batch.action_mask[agent_num]
        for epoch in range(self.n_epochs):

            # with torch.no_grad():
            #     gar = 0
            #     if batch.global_auxiliary_rewards is not None:
            #         gar = batch.global_auxiliary_rewards.unsqueeze(-1)
            #     V_next = self.critic(batch.obs_[agent_num])
            #     # print(
            #     #    f"V_next: {V_next.shape}, batch.global_rewards: {batch.global_rewards.unsqueeze(-1).shape}, obs_: {batch.obs_[agent_num].shape}"
            #     # )
            #     V_targets = (
            #         batch.global_rewards.unsqueeze(-1)
            #         + gar
            #         + (self.gamma * (1 - batch.terminated.unsqueeze(-1)) * V_next)
            #     )
            V_current = self.critic(batch.obs[agent_num])
            loss = (V_current - G).square().mean()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            critic_loss = loss.item()
            print(f"critic_loss: {critic_loss}")

            act, probs, log_probs = self.actor.evaluate(
                batch.obs[agent_num], action_mask=action_mask
            )
            dist = Categorical(probs)
            dist_entropy = dist.entropy()
            selected_log_probs = torch.gather(
                input=log_probs,
                dim=-1,
                index=batch.discrete_actions[agent_num],  # act.unsqueeze(-1)
            )
            ratios = torch.exp(selected_log_probs - batch.discrete_log_probs[agent_num])
            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            actor_loss = (
                -self.policy_loss * torch.min(surr1, surr2).mean()
                + self.entropy_loss * dist_entropy.mean()
            )
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            avg_actor_loss += actor_loss.item()
            print(f"actor_loss: {actor_loss.item()}")

        avg_actor_loss /= self.n_epochs

        return avg_actor_loss, critic_loss

    def save(self, checkpoint_path):
        print("Save not implemeted")

    def load(self, checkpoint_path):
        print("Load not implemented")
