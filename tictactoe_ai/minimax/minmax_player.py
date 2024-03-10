from jax import numpy as jnp, random

from tictactoe_ai.gamerules.types import GameState


def get_action(optimal_play, game: GameState, rng_key):
    board = (game.board + 1).flatten()
    action_values = optimal_play[*board]

    action_mask = board == 1
    action_values *= -game.active_player
    action_values = jnp.where(action_mask, action_values, jnp.inf)

    best_actions = action_values == action_values.min()
    count = jnp.count_nonzero(best_actions)
    probs = best_actions / count

    return random.choice(rng_key, jnp.arange(9), p=probs)
