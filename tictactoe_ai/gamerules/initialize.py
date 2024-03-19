from jax import numpy as jnp

from .over import ONGOING
from .types import GameState


def initialize_game() -> GameState:
    return GameState(
        board=jnp.zeros((3, 3), dtype=jnp.int8),
        active_player=jnp.int8(1),
        over_result=ONGOING,
    )


def initialize_n_games(n: int) -> GameState:
    return GameState(
        board=jnp.zeros((n, 3, 3), dtype=jnp.int8),
        active_player=jnp.ones(n, dtype=jnp.int8),
        over_result=jnp.full(n, ONGOING),
    )
