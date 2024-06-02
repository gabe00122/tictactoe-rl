from jax import numpy as jnp
from flax import linen as nn


def embed(num_embeddings: int, features: int):
    return nn.Embed(
        num_embeddings=num_embeddings,
        features=features,
        embedding_init=nn.initializers.variance_scaling(
            features ** -(1 / 2), "fan_in", "normal", out_axis=0
        ),
    )


class ActorCriticModel(nn.Module):
    body: nn.Module
    actor_head: nn.Module
    critic_head: nn.Module

    @nn.compact
    def __call__(self, inputs, action_mask):
        input_mask = inputs == 1  # empty position
        print(inputs)
        token_embeds = embed(5, 32)(
            inputs
        )  # 0-2 board positions 3,4 our turn or opponents
        position_embeds = embed(10, 32)(
            jnp.arange(0, 10)
        )  # 0-8 board, 9 is the turn token
        embeds = token_embeds + position_embeds

        x = self.body(embeds, input_mask)

        actor_logits = self.actor_head(x, action_mask)
        critic_value = self.critic_head(x)

        return actor_logits, critic_value

    def __hash__(self):
        return id(self)
