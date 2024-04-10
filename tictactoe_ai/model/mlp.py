from jax import numpy as jnp
from flax import linen as nn
from typing import Sequence
from .activation import mish


class FixupMlpBody(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        if len(self.features) == 0:
            return x

        fixup_scale = (len(self.features) - 1) ** (-0.5)
        fixup_init = nn.initializers.variance_scaling(
            scale=2.0 * fixup_scale, mode="fan_in", distribution="truncated_normal"
        )

        x = nn.Dense(self.features[0], kernel_init=nn.initializers.he_normal())(x)
        x = mish(x)

        for i, feat in enumerate(self.features[1:]):
            r_x = x

            fixup_bias_a_1 = self.param(
                f"fixup_bias_a_1_{i}", nn.initializers.zeros, (1,)
            )
            x = nn.Dense(feat, name=f"layer_a_{i}", kernel_init=fixup_init)(
                x + fixup_bias_a_1
            )

            fixup_bias_a_2_ = self.param(
                f"fixup_bias_a_2_{i}", nn.initializers.zeros, (1,)
            )
            x = mish(x + fixup_bias_a_2_)

            fixup_bias_b_1 = self.param(
                f"fixup_bias_b_1_{i}", nn.initializers.zeros, (1,)
            )
            x = nn.Dense(
                feat, name=f"layer gate {i}", kernel_init=nn.initializers.zeros
            )(x + fixup_bias_b_1)

            fixup_multiplier = self.param(
                f"fixup multiplier {i}", nn.initializers.ones, (1,)
            )
            fixup_bias_b_2 = self.param(
                f"fixup_bias_b_2_{i}", nn.initializers.zeros, (1,)
            )

            x = (x * fixup_multiplier + fixup_bias_b_2) + r_x
            x = mish(x)

        return x


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
            kernel_init=nn.initializers.zeros,
        )(inputs)

        actor_logits = jnp.where(mask, actor_logits, -jnp.inf)
        return actor_logits


class CriticHead(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        value = nn.Dense(
            1,
            name="Critic Head",
            kernel_init=nn.initializers.zeros,
        )(inputs)
        return jnp.squeeze(value)
