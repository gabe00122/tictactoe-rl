import jax
from jax import numpy as jnp
from jaxtyping import Float, Bool, Scalar
from functools import partial
from ..gamerules.types import GameState
from ..gamerules.over import ONGOING


# TODO: this needs to be updated for the new rewards
@partial(jax.vmap, in_axes=(0, None))
def get_reward(state: GameState, player: int) -> Float[Scalar, ""]:
    result = state.over_result

    if player == 1:
        rewards = jnp.array([0, -1, 0, 1], dtype=jnp.float32)
    else:
        rewards = jnp.array([0, 1, 0, -1], dtype=jnp.float32)

    return rewards[result]


@jax.vmap
def get_done(state: GameState) -> Bool[Scalar, ""]:
    return state.over_result != ONGOING
