import jax
from jax import numpy as jnp
from jaxtyping import Float, Bool, Scalar
from functools import partial
from .gamerules.types import GameState


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
    return jnp.not_equal(state.over_result, 0)
