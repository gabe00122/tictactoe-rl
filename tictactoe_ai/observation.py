import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Bool
from .gamerules.types import GameState


def get_beforestate_observation(state: GameState) -> Float[Array, "9 3"]:
    def on_over():
        return jnp.zeros((9 * 3))

    def on_not_over():
        num_classes = 3

        board = state["board"] + 1
        board = jax.lax.cond(
            state["active_player"] == -1,
            lambda: board,
            lambda: num_classes - board,
        )

        return jax.nn.one_hot(board, num_classes).flatten()

    return jax.lax.cond(state["over_result"]["is_over"], on_over, on_not_over)


def get_afterstate_observation(state: GameState) -> Float[Array, "9 3"]:
    num_classes = 3

    board = state["board"] + 1
    board = jax.lax.cond(
        state["active_player"] != -1,
        lambda: board,
        lambda: num_classes - board,
    )

    return jax.nn.one_hot(board, num_classes).flatten()


def get_available_actions(state: GameState) -> Bool[Array, "9"]:
    return jnp.equal(state["board"].flatten(), 0)
