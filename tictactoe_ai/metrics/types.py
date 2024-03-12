from typing import TypedDict
from jaxtyping import Scalar, Float


class Metrics(TypedDict):
    state_value: Float[Scalar, ""]
    td_error: Float[Scalar, ""]
    actor_loss: Float[Scalar, ""]
    critic_loss: Float[Scalar, ""]
    entropy: Float[Scalar, ""]
