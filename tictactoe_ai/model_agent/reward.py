import jax
from jax import numpy as jnp
from jaxtyping import Float32, Bool, Scalar, Array
from functools import partial
from ..gamerules.types import GameState
from ..gamerules.over import ONGOING, WON


@partial(jax.vmap, in_axes=(0, 0))
def get_reward(state: GameState, active_player: Bool[Array, "vec"]) -> Float32[Scalar, ""]:
    return jax.lax.cond(
        state.over_result == WON,
        lambda: jax.lax.cond(active_player, lambda: jnp.float32(1), lambda: jnp.float32(-1)),
        lambda: jnp.float32(0),
    )


@jax.vmap
def get_done(state: GameState) -> Bool[Scalar, ""]:
    return state.over_result != ONGOING
