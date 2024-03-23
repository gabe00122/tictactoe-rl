from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pathlib import Path
from jaxtyping import Array, Int8, Key, Float32, PRNGKeyArray, Bool

from .gamerules.types import GameState
from .metrics import Metrics

S = TypeVar("S")


class Agent(ABC, Generic[S]):
    @abstractmethod
    def initialize(self, rng_key: PRNGKeyArray, env_num: int) -> S:
        pass

    @abstractmethod
    def act(
        self, agent_state: S, game_state: GameState, rng_key: Key[Array, ""]
    ) -> tuple[Int8[Array, ""], Float32[Array, "9"]]:
        # returns an action
        pass

    @abstractmethod
    def learn(
        self,
        params: S,
        game_states: GameState,
        actions: Int8[Array, "vec"],
        next_obs: GameState,
        active_agents: Int8[Array, "vec"],
    ) -> tuple[S, Metrics]:
        pass

    @abstractmethod
    def load(self, path: Path) -> S:
        pass

    @abstractmethod
    def save(self, path: Path, state: S):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
