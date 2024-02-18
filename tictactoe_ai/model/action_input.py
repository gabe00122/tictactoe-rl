from typing import TypedDict
from jaxtyping import Array, Float32, Bool
import jax
from jax import numpy as jnp
from ..gamerules.types.state import GameState


class ActionInput(TypedDict):
    obs: Float32[Array, "9 3"]
    avalible_actions: Bool[Array, "9"]


def action_input(state: GameState) -> ActionInput:
    board = state['board'].flatten()
    
    return {
        'obs': jax.nn.one_hot(board + 1, 3),
        'avalible_actions': jnp.equal(board, 0),
    }
