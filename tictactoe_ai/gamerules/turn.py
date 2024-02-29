from .types.state import GameState
from .over import check_gameover
from .initialize import initialize_game
import jax
from jax import numpy as jnp
from jaxtyping import Scalar, Int8


def turn(state: GameState, action: Int8[Scalar, ""]) -> GameState:
    board = state["board"]
    active_player = state["active_player"]

    x_pos = action % 3
    y_pos = action // 3
    board = board.at[y_pos, x_pos].set(active_player)
    active_player = -active_player

    return {
        "board": board,
        "active_player": active_player,
        "over_result": check_gameover(board),
    }


def reset_if_done(state: GameState):
    return jax.lax.cond(
        state["over_result"]["is_over"],
        lambda: initialize_game() | {"active_player": state["active_player"]},
        lambda: state,
    )
