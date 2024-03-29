from .types import GameState
from .over import check_game_over, ONGOING
from .initialize import initialize_game
import jax
from jaxtyping import ScalarLike, Int, PRNGKeyArray


def turn(state: GameState, action: Int[ScalarLike, ""]) -> GameState:
    board = state.board
    active_player = state.active_player

    x_pos = action % 3
    y_pos = action // 3
    board = board.at[y_pos, x_pos].set(active_player)

    return GameState(
        board=board,
        active_player=-active_player,
        over_result=check_game_over(board, active_player),
    )


def reset_if_done(state: GameState) -> GameState:
    return jax.lax.cond(
        state.over_result != ONGOING,
        lambda: initialize_game(),
        lambda: state,
    )
