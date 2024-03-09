import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Bool

from .gamerules.types import GameState


def get_observation(state: GameState, player: int) -> Float[Array, "27"]:
    num_classes = 3

    board = state["board"] + 1
    board = board if player == -1 else 2 - board
    return jax.nn.one_hot(board, num_classes).flatten()


get_observation_vec = jax.vmap(get_observation, (0, None))



def get_available_actions(state: GameState) -> Bool[Array, "9"]:
    return jnp.equal(state["board"].flatten(), 0)


get_available_actions_vec = jax.vmap(get_available_actions)
