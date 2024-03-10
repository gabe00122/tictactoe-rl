from typing import TypedDict
from jaxtyping import Float, Array


class Metrics(TypedDict):
    state_value: Float[Array, "buffer"]
    td_error: Float[Array, "buffer"]
    actor_loss: Float[Array, "buffer"]
    critic_loss: Float[Array, "buffer"]
    entropy: Float[Array, "buffer"]
