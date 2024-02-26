import jax
from jax import numpy as jnp
from jaxtyping import Int, Float, Bool, Scalar
from functools import partial
from .gamerules.types import GameState


@partial(jax.vmap, (0, None))
def get_reward(state: GameState, player: Int[Scalar, ""]) -> Float[Scalar, ""]:
    result = state["over_result"]
    is_over = result["is_over"]
    winner = result["winner"]

    return jax.lax.cond(
        is_over,
        lambda: jax.lax.cond(
            player, lambda: winner, lambda: -winner
        ).astype(jnp.float32),
        lambda: jnp.float32(0),
    )


@jax.vmap
def get_done(state: GameState) -> Bool[Scalar, ""]:
    return state["over_result"]["is_over"]
