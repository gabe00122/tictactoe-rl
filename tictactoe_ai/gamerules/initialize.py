import jax
from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray, Key, Array
from .types import GameState, VectorizedGameState


def initialize_game(rng_key: PRNGKeyArray) -> GameState:
    return GameState(
        board=jnp.zeros((3, 3), dtype=jnp.int8),
        active_player=random.choice(rng_key, jnp.array([-1, 1], dtype=jnp.int8)),
        over_result=jnp.int8(0),
    )


def initialize_n_games(rng_keys: Key[Array, "vec"]) -> VectorizedGameState:  # vectorized GameState
    return jax.vmap(initialize_game, 0)(rng_keys)
