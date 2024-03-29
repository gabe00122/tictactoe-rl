from pathlib import Path
from jax import numpy as jnp, random
from typing import NamedTuple
from jaxtyping import Array, Int8, PRNGKeyArray, Key, Float32, Int, Scalar

from tictactoe_ai.gamerules.types import GameState
from tictactoe_ai.agent import Agent
from tictactoe_ai.metrics import Metrics, empty_metrics


class MinmaxState(NamedTuple):
    state_actions: Int8[Array, "3 3 3 3 3 3 3 3 3 9"]


class MinmaxAgent(Agent[MinmaxState]):
    def initialize(self, rng_key: PRNGKeyArray, env_num: int) -> MinmaxState:
        return MinmaxState(jnp.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3, 9), dtype=jnp.int8))

    def act(
        self,
        agent_state: MinmaxState,
        game: GameState,
        rng_key: Key[Array, ""],
    ) -> tuple[Int8[Array, ""], Float32[Array, "9"]]:
        board = (game.board + 1).flatten()
        action_values = agent_state.state_actions[*board]

        action_mask = board == 1
        action_values *= game.active_player
        action_values = jnp.where(action_mask, action_values, -jnp.inf)

        best_actions = action_values == action_values.max()
        count = jnp.count_nonzero(best_actions)
        probs = best_actions / count

        return random.choice(rng_key, 9, p=probs), probs

    def learn(
        self,
        params: MinmaxState,
        game_states: GameState,
        actions: Int8[Array, "vec"],
        next_obs: GameState,
        active_agents: Int8[Array, "vec"],
        step: Int[Scalar, ""],
        total_steps: Int[Scalar, ""],
    ) -> tuple[MinmaxState, Metrics]:
        return params, empty_metrics()

    def save(self, path: Path, state: MinmaxState):
        return jnp.save(path, state.state_actions)

    def load(self, path: str | Path) -> MinmaxState:
        return MinmaxState(jnp.load(path))

    def get_name(self) -> str:
        return "MinMax"
