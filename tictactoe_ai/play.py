import jax
from jax import numpy as jnp, random
from tictactoe_ai.gamerules.initialize import initialize_game
from tictactoe_ai.gamerules.types.state import GameState
from tictactoe_ai.gamerules.turn import turn
from tictactoe_ai.display import display
from tictactoe_ai.random_player import get_random_move


initialize_game = jax.jit(initialize_game)
get_random_move = jax.jit(get_random_move)
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
        key, move_key = random.split(key)
        move = get_random_move(state, key)
        state = turn(state, move)
        display(state)


def get_human_move():
    x = int(input("X: "))
    y = int(input("Y: "))
    return jnp.int8(x + y * 3)


if __name__ == "__main__":
    play()
