import jax
from jax import numpy as jnp
from jaxtyping import Scalar, Bool, Int8, Array

# result key: ongoing, o won, tied, x won


def check_gameover(board: Int8[Array, "3 3"]) -> Int8[Scalar, ""]:
    players = jnp.array([-1, 1], dtype=jnp.int8)

    v_check_player_won = jax.vmap(check_player_won, in_axes=(None, 0))
    winners = v_check_player_won(board, players)

    is_winner = jnp.any(winners)

    def some_winner():
        winner = jnp.argmax(winners)
        return jax.lax.cond(winner == 0, lambda: jnp.int8(1), lambda: jnp.int8(3))

    def no_winner():
        no_space_left = check_no_space_left(board)
        return jax.lax.cond(no_space_left, lambda: jnp.int8(2), lambda: jnp.int8(0))

    return jax.lax.cond(is_winner, some_winner, no_winner)


def check_player_won(
    board: Int8[Array, "3 3"], player: Int8[Scalar, ""]
) -> Bool[Scalar, ""]:
    mask = board == player
    out = mask.all(0).any() | mask.all(1).any()
    out |= jnp.diag(mask).all() | jnp.diag(mask[:, ::-1]).all()
    return out


def check_no_space_left(board: Int8[Array, "3 3"]) -> Bool[Scalar, ""]:
    return jnp.all(board != 0)
