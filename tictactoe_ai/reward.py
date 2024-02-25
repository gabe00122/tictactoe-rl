import jax
from jax import numpy as jnp
from jaxtyping import Float, Bool, Scalar
from .gamerules.types import GameState


@jax.vmap
def get_reward(state: GameState) -> Float[Scalar, ""]:
    result = state["over_result"]
    is_over = result["is_over"]
    winner = result["winner"]
    previous_active_player = -state["active_player"]

    return jax.lax.cond(
        is_over,
        lambda: jax.lax.cond(
            previous_active_player == 1, lambda: winner, lambda: -winner
        ).astype(jnp.float32),
        lambda: jnp.float32(0),
    )


@jax.vmap
def get_done(state: GameState) -> Bool[Scalar, ""]:
    return state["over_result"]["is_over"]
