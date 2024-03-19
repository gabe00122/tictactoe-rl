import jax
from jax import numpy as jnp
from jaxtyping import Scalar, Bool, Int8, Array

# result key: ongoing, draw, won
ONGOING = jnp.int8(-1)
DRAW = jnp.int8(0)
WON = jnp.int8(1)


def check_game_over(board, active_player) -> Int8[Scalar, ""]:
    is_winner = check_player_won(board, active_player)
    is_draw = check_no_space_left(board)

    return jax.lax.cond(
        is_winner,
        lambda: WON,
        lambda: jax.lax.cond(is_draw, lambda: DRAW, lambda: ONGOING),
    )


def check_player_won(
    board: Int8[Array, "3 3"], player: Int8[Scalar, ""]
) -> Bool[Scalar, ""]:
    mask = board == player
    out = mask.all(0).any() | mask.all(1).any()
    out |= jnp.diag(mask).all() | jnp.diag(mask[:, ::-1]).all()
    return out


def check_no_space_left(board: Int8[Array, "3 3"]) -> Bool[Scalar, ""]:
    return jnp.all(board != 0)
