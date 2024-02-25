from flax import linen as nn
from typing import Sequence
from .activation import mish


class Mlp(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for i, feat in enumerate(self.features):
            last_feat = i == len(self.features) - 1

            x = nn.Dense(
                feat,
                name=f"Layer {i}",
                kernel_init=nn.initializers.he_normal(),
            )(x)
            if not last_feat:
                x = mish(x)
        return x
