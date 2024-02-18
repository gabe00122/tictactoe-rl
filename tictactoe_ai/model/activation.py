from jax import numpy as jnp
from flax import linen as nn


def mish(x: jnp.ndarray):
    return x * jnp.tanh(nn.softplus(x))
