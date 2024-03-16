from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pathlib import Path
from jaxtyping import Array, Int8, Key, Float32, PRNGKeyArray

from .gamerules.types import VectorizedGameState
from .metrics import Metrics

S = TypeVar('S')


class Agent(ABC, Generic[S]):
    @abstractmethod
    def initialize(self, rng_key: PRNGKeyArray) -> S:
        pass

    @abstractmethod
    def act(self, agent_state: S, game_states: VectorizedGameState, rng_keys: Key[Array, "vec"]) -> Int8[Array, "vec"]:
        # returns an action
        pass

    @abstractmethod
    def learn(
        self,
        params: S,
        game_states: VectorizedGameState,
        actions: Int8[Array, "vec"],
        rewards: Float32[Array, "vec"],
        next_obs: VectorizedGameState
    ) -> tuple[S, Metrics]:
        pass

    @abstractmethod
    def load(self, path: Path) -> S:
        pass

    @abstractmethod
    def save(self, path: Path, state: S):
        pass
