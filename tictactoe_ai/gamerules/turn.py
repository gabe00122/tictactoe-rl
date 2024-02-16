from .types.state import GameState
from .over import check_gameover
import jax
from jaxtyping import Scalar, Int8


def turn(state: GameState, action: Int8[Scalar, ""]) -> GameState:
    board = state['board']
    active_player = state['active_player']
    
    x_pos = action % 3
    y_pos = action // 3
    board = board.at[y_pos, x_pos].set(active_player)
    active_player = jax.lax.cond(active_player == 1, lambda: -1, lambda: 1)
    
    return {
        'board': board,
        'active_player': active_player,
        'over_result': check_gameover(board),
    }
