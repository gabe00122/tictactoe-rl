from jax import numpy as jnp
from .types.state import GameState


def initalize_game() -> GameState:
    return {
        'board': jnp.zeros((3, 3), dtype=jnp.int8),
        'active_player': jnp.int8(1),
        'over_result': {
            'is_over': jnp.bool(False),
            'winner': jnp.int8(0)
        },
    }
