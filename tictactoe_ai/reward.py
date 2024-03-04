import jax
from jax import numpy as jnp
from jaxtyping import Int, Float, Bool, Scalar
from functools import partial
from .gamerules.types import GameState


@partial(jax.vmap, in_axes=(0, None))
def get_reward(state: GameState, player: Int[Scalar, ""]) -> Float[Scalar, ""]:
    result = state["over_result"]

    rewards = jnp.array([0, -1, 0, 1], dtype=jnp.float32)
    reward = rewards[result.game_state]
    return jax.lax.cond(player == 1, lambda: reward, lambda: -reward)


@jax.vmap
def get_done(state: GameState) -> Bool[Scalar, ""]:
    return jnp.not_equal(state["over_result"].game_state, 0)
