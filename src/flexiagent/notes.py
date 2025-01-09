old_cont_logits, old_disc_logits = self.actor(batch.obs[agent_num, mini_batch_indices])
cdist = torch.distributions.Normal(
    loc=old_cont_logits,
    scale=torch.exp(self.actor_logstd.expand_as(old_cont_logits)),
)
old_cont_logprobs = cdist.log_prob(
    batch.continuous_actions[agent_num, mini_batch_indices]
)
