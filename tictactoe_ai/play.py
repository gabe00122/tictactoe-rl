import jax
from jax import numpy as jnp, random
from tictactoe_ai.gamerules.initialize import initialize_game
from tictactoe_ai.gamerules.types.state import GameState
from tictactoe_ai.gamerules.turn import turn
from tictactoe_ai.display import display

initalize_game = jax.jit(initialize_game)
turn = jax.jit(turn)


def play():
    key = random.PRNGKey(434)

    state: GameState = initialize_game()
    display(state)
    while not state["over_result"]["is_over"]:
        # X's
        move = get_human_move()
        state = turn(state, move)

        # O's
        move, key = get_random_move(state, key)
        state = turn(state, move)
        display(state)


@jax.jit
def get_random_move(state, key):
    board = state["board"]
    avalible_moves = board.reshape((9,)) == 0
    count = jnp.count_nonzero(avalible_moves)
    probs = avalible_moves / count
    key, choice_key = random.split(key)

    return random.choice(choice_key, jnp.arange(9), p=probs), key


def get_human_move():
    x = int(input("X: "))
    y = int(input("Y: "))
    return jnp.int8(x + y * 3)


if __name__ == "__main__":
    play()
