import jax

from tictactoe_ai.gamerules.types import GameState
from tictactoe_ai.gamerules.turn import turn
from tictactoe_ai.gamerules.initialize import initialize_game
from typing import NamedTuple
from jax import numpy as jnp


turn = jax.jit(turn)
counter = 0


class Results(NamedTuple):
    o_wins: int
    ties: int
    x_wins: int


def minmax(state: GameState) -> Results:
    global counter
    counter += 1
    if counter % 1000 == 0:
        print(counter)

    if state['over_result'] != 0:
        return Results(
            int(state['over_result'] == 1),
            int(state['over_result'] == 2),
            int(state['over_result'] == 3),
        )
    else:
        board = state['board'].flatten().tolist()
        o_wins = 0
        ties = 0
        x_wins = 0

        for action, cell in enumerate(board):
            if cell == 0:
                result = minmax(turn(state, jnp.int8(action)))
                o_wins += result.o_wins
                ties += result.ties
                x_wins += result.x_wins

        return Results(o_wins, ties, x_wins)


def main():
    results = minmax(initialize_game())
    print(results)


if __name__ == '__main__':
    main()
