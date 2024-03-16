from jax import numpy as jnp
from typing import TypedDict
from jaxtyping import Scalar, Float


class Metrics(TypedDict):
    state_value: Float[Scalar, ""]
    td_error: Float[Scalar, ""]
    actor_loss: Float[Scalar, ""]
    critic_loss: Float[Scalar, ""]
    entropy: Float[Scalar, ""]


def empty_metrics() -> Metrics:
    return Metrics(
        state_value=jnp.float32(0),
        td_error=jnp.float32(0),
        actor_loss=jnp.float32(0),
        critic_loss=jnp.float32(0),
        entropy=jnp.float32(0),
    )