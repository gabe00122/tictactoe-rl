from pathlib import Path

from tictactoe_ai.gamerules.types import GameState
from jaxtyping import PRNGKeyArray, Int8, Float32, Key, Bool
from jax import numpy as jnp, random, Array
from tictactoe_ai.agent import Agent
from tictactoe_ai.metrics import Metrics, empty_metrics


class RandomAgent(Agent[None]):
    def initialize(self, rng_key: PRNGKeyArray, env_num: int) -> None:
        return

    def act(
        self, agent_state: None, game: GameState, rng_key: Key[Array, ""]
    ) -> Int8[Array, ""]:
        board = game.board
        available_moves = board.flatten() == 0

        count = jnp.count_nonzero(available_moves)
        probs = available_moves / count

        return random.choice(rng_key, jnp.arange(9), p=probs)

    def learn(
        self,
        params: None,
        game_states: GameState,
        actions: Int8[Array, "vec"],
        next_obs: GameState,
        active_agents: Int8[Array, "vec"],
    ) -> tuple[None, Metrics]:
        return None, empty_metrics()

    def save(self, path: Path, state: None):
        pass

    def load(self, path: Path) -> None:
        return
