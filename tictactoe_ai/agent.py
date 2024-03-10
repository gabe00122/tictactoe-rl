from abc import ABC, abstractmethod
from typing import TypeVar, Generic, NamedTuple
from jaxtyping import PRNGKeyArray, Array, Int8, Key
from .gamerules.types import VectorizedGameState
from .metrics import Metrics

S = TypeVar('S')

class Agent(ABC, Generic[S]):

    @abstractmethod
    def initialize(self) -> S:
        pass

    @abstractmethod
    def act(self, agent_state: S, game_states: VectorizedGameState, rng_keys: Key[Array, "vec"]) -> Int8[Array, "vec"]:
        """
        Vectorized action
        :param agent_state:
        :param game_states:
        :param rng_keys:
        :return:
        """
        pass
