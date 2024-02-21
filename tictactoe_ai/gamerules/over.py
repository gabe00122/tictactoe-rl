import jax
from jax import numpy as jnp
from .types.over import OverResult
from jaxtyping import Scalar, Bool, Int8, Array


def check_gameover(board: Int8[Array, "3 3"]) -> OverResult:
    players = jnp.array([-1, 1], dtype=jnp.int8)
    
    v_check_player_won = jax.vmap(check_player_won, in_axes=(None, 0))
    winners = v_check_player_won(board, players)
    
    is_winner = jnp.any(winners)
    
    def some_winner():
        winner = jnp.argmax(winners)
        winner = jax.lax.cond(winner == 0, lambda: jnp.int8(-1), lambda: jnp.int8(1))
        return {
            'is_over': True,
            'winner': winner,
        }
    
    def no_winner():
        no_space_left = check_no_space_left(board)
        return {
            'is_over': no_space_left,
            'winner': jnp.int8(0),
        }
    
    return jax.lax.cond(is_winner, some_winner, no_winner)


def check_player_won(board: Int8[Array, "3 3"], player: Int8[Scalar, ""]) -> Bool[Scalar, ""]:
    mask = board == player
    out = mask.all(0).any() | mask.all(1).any()
    out |= jnp.diag(mask).all() | jnp.diag(mask[:, ::-1]).all()
    return out


def check_no_space_left(board: Int8[Array, "3 3"]) -> Bool[Scalar, ""]:
    return jnp.all(board != 0)
    