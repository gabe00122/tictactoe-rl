import jax
import numpy as np
from jax import numpy as jnp

from tictactoe_ai.gamerules.initialize import initialize_game
from tictactoe_ai.gamerules.turn import turn
from tictactoe_ai.gamerules.types import GameState

turn = jax.jit(turn)


# key: o_wins, ties, x_wins
EMPTY = -2
O_PLAYER = -1
X_PLAYER = 1

def minmax(states, state_actions, state: GameState):
    board = state.board.flatten().tolist()
    index = [x + 1 for x in board]

    if states[*index] != EMPTY:
        return states[*index]

    if state.over_result != 0:
        value = state.over_result - 2
    else:
        first = True
        value = 0

        for action, cell in enumerate(board):
            if cell == 0:
                child_value = minmax(states, state_actions, turn(state, jnp.int8(action)))
                state_actions[*index, action] = child_value

                if first:
                    value = child_value
                    first = False
                elif (state.active_player == O_PLAYER and child_value < value) or (state.active_player == X_PLAYER and child_value > value):
                    value = child_value

    states[*index] = value
    return value


def main():
    states = np.full(
        (
            3, 3, 3,
            3, 3, 3,
            3, 3, 3,
        ),
        EMPTY,
        dtype=np.int8
    )
    state_actions = np.zeros(
        (
            3, 3, 3,
            3, 3, 3,
            3, 3, 3,
            9,
        ),
        dtype=np.int8,
    )

    minmax(states, state_actions, initialize_game())
    np.save("./optimal_play.npy", state_actions)


if __name__ == "__main__":
    main()
