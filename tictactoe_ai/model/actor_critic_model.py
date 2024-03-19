from flax import linen as nn


class ActorCriticModel(nn.Module):
    body: nn.Module
    actor_head: nn.Module
    critic_head: nn.Module

    @nn.compact
    def __call__(self, inputs, mask):
        x = inputs
        x = self.body(x)

        actor_logits = self.actor_head(x, mask)
        critic_value = self.critic_head(x)

        return actor_logits, critic_value

    def __hash__(self):
        return id(self)
