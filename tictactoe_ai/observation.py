import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Bool
from .gamerules.types import GameState


def get_beforestate_observation(state: GameState) -> Float[Array, "9 3"]:
    num_classes = 3
    
    board = state["board"].flatten() + 1
    board = jax.lax.cond(
        state["active_player"] == -1,
        lambda: board,
        lambda: num_classes - board,
    )
    
    return jax.nn.one_hot(board, num_classes)


def get_afterstate_observation(state: GameState) -> Float[Array, "9 3"]:
    num_classes = 3

    board = state["board"].flatten() + 1
    board = jax.lax.cond(
        state["active_player"] != -1,
        lambda: board,
        lambda: num_classes - board,
    )

    return jax.nn.one_hot(board, num_classes)


def get_available_actions(state: GameState) -> Bool[Array, "9"]:
    return jnp.equal(state["board"], 0)
