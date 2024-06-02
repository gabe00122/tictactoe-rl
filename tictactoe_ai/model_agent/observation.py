import jax
from jax import numpy as jnp
from jaxtyping import Array, Scalar, Int8, Float, Bool

from ..gamerules.types import GameState


def get_observation(state: GameState, player: Int8[Scalar, ""]) -> Float[Array, "27"]:
    num_classes = 3

    board = state.board + 1
    board = jax.lax.cond(player == -1, lambda: board, lambda: 2 - board)

    board_encoding = jnp.ravel(board)  # jax.nn.one_hot(board, num_classes).flatten()
    is_turn = jax.lax.cond(
        state.active_player == player,
        lambda: jnp.array([3], dtype=jnp.int8),
        lambda: jnp.array([3], dtype=jnp.int8),
    )

    temp = jnp.concatenate([board_encoding, is_turn], dtype=jnp.int8)
    return temp


get_observation_vec = jax.vmap(get_observation, (0, 0))


def get_available_actions(state: GameState) -> Bool[Array, "9"]:
    return jnp.equal(state.board.flatten(), 0)


get_available_actions_vec = jax.vmap(get_available_actions)
