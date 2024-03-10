import jax
import numpy as np
from jax import numpy as jnp

from tictactoe_ai.gamerules.initialize import initialize_game
from tictactoe_ai.gamerules.turn import turn
from tictactoe_ai.gamerules.types import GameState

turn = jax.jit(turn)
counter = 0


# key: o_wins, ties, x_wins = 0


def minmax(data, state: GameState):
    global counter
    counter += 1
    if counter % 1000 == 0:
        print(counter)

    over = state.over_result
    if over != 0:
        return over - 2
    else:
        board = state.board.flatten().tolist()
        index = [x + 1 for x in board]
        first = True
        value = 0

        for action, cell in enumerate(board):
            if cell == 0:
                child_value = minmax(data, turn(state, jnp.int8(action)))
                data[*index, action] = child_value

                if first:
                    value = child_value
                    first = False
                elif (state.active_player == -1 and child_value < value) or (
                    state.active_player == 1 and child_value > value
                ):
                    value = child_value

        return value


def main():
    data = np.zeros(
        (
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            9,
        ),
        dtype=np.int8,
    )

    results = minmax(data, initialize_game())
    print(results)
    np.save("./optimal_play.npy", data)


if __name__ == "__main__":
    main()
