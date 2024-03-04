from .gamerules.types import GameState
from jaxtyping import PRNGKeyArray
from jax import numpy as jnp, random




def get_random_move(state: GameState, rng_key: PRNGKeyArray):
    board = state["board"]
    available_moves = board.flatten() == 0

    count = jnp.count_nonzero(available_moves)
    probs = available_moves / count

    return random.choice(rng_key, jnp.arange(9), p=probs)
