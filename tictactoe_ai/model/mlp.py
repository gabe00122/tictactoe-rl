from jax import numpy as jnp
from flax import linen as nn
from typing import Sequence
from .activation import mish


class MlpBody(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for i, feat in enumerate(self.features):
            x = nn.Dense(
                feat,
                name=f"Layer {i}",
                kernel_init=nn.initializers.he_normal(),
            )(x)
            x = mish(x)
        return x


class ActorHead(nn.Module):
    actions: int

    @nn.compact
    def __call__(self, inputs, mask):
        actor_logits = nn.Dense(
            self.actions,
            name="Actor Head",
            kernel_init=nn.initializers.variance_scaling(
                2.0, "fan_in", "truncated_normal"
            ),
        )(inputs)

        actor_logits = jnp.where(mask, actor_logits, -jnp.inf)
        return actor_logits


class CriticHead(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        value = nn.Dense(
            1, name="Critic Head", kernel_init=nn.initializers.he_normal()
        )(inputs)
        return jnp.squeeze(value)
