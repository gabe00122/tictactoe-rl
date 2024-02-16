import jax
from jax import numpy as jnp, random
from .gamerules.initialize import initalize_game
from .gamerules.types.state import GameState
from .gamerules.turn import turn
from .display import display

initalize_game = jax.jit(initalize_game)
turn = jax.jit(turn)


def play():
    key = random.PRNGKey(434)
    
    state: GameState = initalize_game()
    display(state)
    while not state['over_result']['is_over']:
        move = int(input("Move: "))
        state = turn(state, jnp.int8(move))
        
        board = state["board"]
        avalible = board.reshape((9, )) == 0
        count = jnp.count_nonzero(avalible)
        probs = avalible / count
        key, choice_key = random.split(key)
        
        move = random.choice(choice_key, jnp.arange(9), p=probs)
        state = turn(state, move)
        display(state)


if __name__ == '__main__':
    play()
