import jax
from jax import numpy as jnp
from jaxtyping import Array, Scalar, Int8, Float, Bool

from ..gamerules.types import GameState


def get_observation(state: GameState, player: Int8[Scalar, ""]) -> Float[Array, "27"]:
    num_classes = 3

    board = state.board + 1
    board = jax.lax.cond(player == -1, lambda: board, lambda: 2 - board)

    board_encoding = jax.nn.one_hot(board, num_classes).flatten()
    is_turn = jnp.array([
        state.active_player == player,
        state.active_player != player
    ], jnp.float32)

    return jnp.concatenate([board_encoding, is_turn])


get_observation_vec = jax.vmap(get_observation, (0, 0))


def get_available_actions(state: GameState) -> Bool[Array, "9"]:
    return jnp.equal(state.board.flatten(), 0)


get_available_actions_vec = jax.vmap(get_available_actions)
