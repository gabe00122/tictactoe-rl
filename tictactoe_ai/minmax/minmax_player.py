from pathlib import Path
import jax
from jax import numpy as jnp, random
from typing import NamedTuple
from jaxtyping import Array, Int8, PRNGKeyArray, Key

from tictactoe_ai.gamerules.types import GameState, VectorizedGameState
from tictactoe_ai.agent import Agent
from tictactoe_ai.metrics import Metrics, empty_metrics


def get_action(
    optimal_play: Int8[Array, "3 3 3 3 3 3 3 3 3 9"],
    game: GameState,
    rng_key: PRNGKeyArray,
):
    board = (game.board + 1).flatten()
    action_values = optimal_play[*board]

    action_mask = board == 1
    action_values *= game.active_player
    jax.debug.print("{}", action_values)
    action_values = jnp.where(action_mask, action_values, -jnp.inf)

    best_actions = action_values == action_values.max()
    count = jnp.count_nonzero(best_actions)
    probs = best_actions / count

    return random.choice(rng_key, jnp.arange(9, dtype=jnp.int8), p=probs)


class MinmaxState(NamedTuple):
    state_actions: Int8[Array, "3 3 3 3 3 3 3 3 3 9"]


class MinmaxAgent(Agent[MinmaxState]):
    def initialize(self, rng_key: PRNGKeyArray) -> MinmaxState:
        return MinmaxState(jnp.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3, 9), dtype=jnp.int8))

    def act(
        self,
        agent_state: MinmaxState,
        game_states: VectorizedGameState,
        rng_keys: Key[Array, "vec"],
    ) -> Int8[Array, "vec"]:
        return jax.vmap(get_action, in_axes=(None, 0, 0))(agent_state.state_actions, game_states, rng_keys)  # type: ignore

    def learn(
        self,
        params: MinmaxState,
        game_states: VectorizedGameState,
        actions: Array,
        rewards: Array,
        next_obs: VectorizedGameState,
    ) -> tuple[MinmaxState, Metrics]:
        return params, empty_metrics()

    def save(self, path: Path, state: MinmaxState):
        return jnp.save(path, state.state_actions)

    def load(self, path: Path) -> MinmaxState:
        return MinmaxState(jnp.load(path))
