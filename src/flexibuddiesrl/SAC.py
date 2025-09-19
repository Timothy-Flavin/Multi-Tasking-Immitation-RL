from flexibuddiesrl.Agent import Agent, StochasticActor, ValueS
import torch


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
        log_std_clamp_range=[-3, 5],
        lr=1e-3,
        actor_ratio=0.5,
        gamma=0.99,
        sac_tau=0.005,
        initial_temperature=0.2,
    ):

        action_dim = continuous_action_dim
        if discrete_action_dims is not None:
            action_dim += sum(discrete_action_dims)
        self.lr = lr
        self.actor_ratio = actor_ratio
        self.gamma = gamma
        self.sac_tau = sac_tau
        self.initial_temperature = initial_temperature
        self.device = device

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

        self.critic = ValueS(
            obs_dim=obs_dim + action_dim,
            hidden_dim=hidden_dims,
            device=device,
            activation=activation,
            orthogonal_init=orthogonal_init,
        )

        self.critic_target = ValueS(
            obs_dim=obs_dim + action_dim,
            hidden_dim=hidden_dims,
            device=device,
            activation=activation,
            orthogonal_init=orthogonal_init,
        )

        for param in self.critic.state_dict():
            print("avg")
            # TODO: polyak average critic and target

        # TODO: Set up adam optimizer

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
            "discrete_action": discrete_actions,
            "continuous_action": continuous_actions,
            "discrete_log_prob": discrete_log_probs,
            "continuous_log_prob": continuous_log_probs,
            "value": 0,
            "time": 0,
        }

    def ego_actions(self, observations, action_mask=None) -> dict:
        return {
            "discrete_action": 0,
            "continuous_action": 0,
        }

    def imitation_learn(
        self,
        observations,
        continuous_actions,
        discrete_actions,
        action_mask=None,
        debug=False,
    ) -> dict:
        immitation_metrics = {"critic_loss": 0, "actor_loss": 0, "time": 0}
        return immitation_metrics

    def utility_function(self, observations, actions=None):
        return 0  # Returns the single-agent critic for a single action.
        # If actions are none then V(s)

    def expected_V(self, obs, legal_action) -> torch.Tensor | np.ndarray | float:
        # Return the expected value from our critic
        return 0.0

    def stable_greedy(self, obs, legal_action):
        """
        Sample a greedy action from this agent's target or stable
        policy. For DQN this is argmax(target_Q), for PPO this is
        just like taking a train action which is equal in
        expectation to the current policy.
        """
        print("stable greedy not implemented")
        return None, None

    def reinforcement_learn(
        self, batch, agent_num=0, critic_only=False, debug=False
    ) -> dict:
        obs = batch.__getattr__("obs")
        obs_ = batch.__getattr__("obs_")
        rewards = batch.__getattr__("global_reward")
        discrete_actions = batch.__getattr__("discrete_actions")
        continuous_actions = batch.__getattr__("continuous_actions")

        # TODO: soft actor critic reinforcement learn

        rl_metrics = {
            "critic_loss": 0,
            "d_actor_loss": 0,
            "c_actor_loss": 0,
            "d_entropy": 0,
            "c_entropy": 0,
            "c_std": 0,
        }
        return rl_metrics

    def save(self, checkpoint_path):
        # Save the model in the checkpoint path
        print("Save not implemeted")

    def load(self, checkpoint_path):
        # Save the model from the checkpoint path
        print("Load not implemented")

    def param_count(self) -> tuple[int, int]:
        # First number is the policy param count
        # Second is the critic + policy param counts
        return 0, 0  # train and execute param count
