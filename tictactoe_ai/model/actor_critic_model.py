from flax import linen as nn


class ActorCriticModel(nn.Module):
    body: nn.Module
    actor_neck: nn.Module
    actor_head: nn.Module
    critic_neck: nn.Module
    critic_head: nn.Module

    @nn.compact
    def __call__(self, inputs, mask):
        x = self.body(inputs)

        actor_x = self.actor_neck(x)
        critic_x = self.critic_neck(x)

        actor_logits = self.actor_head(actor_x, mask)
        critic_value = self.critic_head(critic_x)

        return actor_logits, critic_value

    def __hash__(self):
        return id(self)
