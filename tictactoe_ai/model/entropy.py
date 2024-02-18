from jax import numpy as jnp
from jaxtyping import Array, Scalar, Float


def mul_exp(x, logp):
    p = jnp.exp(logp)
    x = jnp.where(p == 0, 0.0, x)
    return x * p


def entropy_loss(action_probs: Float[Array, "actions"]) -> Float[Scalar, ""]:
    return jnp.sum(mul_exp(action_probs, action_probs))
