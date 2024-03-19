from pathlib import Path
from typing import NamedTuple

import orbax.checkpoint as ocp
from jax import random
from jaxtyping import Int8, Float32, Key, PRNGKeyArray, Array

from .observation import get_observation, get_available_actions
from ..agent import Agent
from ..gamerules import GameState
from ..metrics import Metrics
from ..model.actor_critic import ActorCritic, TrainingState


class ActorCriticState(NamedTuple):
    training_state: TrainingState


class ActorCriticAgent(Agent[ActorCriticState]):
    def __init__(self, model: ActorCritic):
        self.model = model

    def initialize(self, rng_key: PRNGKeyArray) -> ActorCriticState:
        training_state = self.model.init(rng_key)
        return ActorCriticState(training_state)

    def act(self, agent_state: ActorCriticState, game_state: GameState, rng_key: Key[Array, ""]) -> Int8[Array, ""]:
        obs = get_observation(game_state, game_state.active_player)
        available_actions = get_available_actions(game_state)
        action = self.model.act(agent_state.training_state, obs, available_actions, rng_key)
        return action

    def learn(self, agent_state: ActorCriticState, game_states: GameState, actions: Int8[Array, "vec"], rewards: Float32[Array, "vec"],
              next_obs: GameState) -> tuple[ActorCriticState, Metrics]:
        pass

    def load(self, path: Path) -> ActorCriticState:
        random_params = self.model.init(random.PRNGKey(0))

        checkpointer = ocp.PyTreeCheckpointer()
        restore_args = ocp.checkpoint_utils.construct_restore_args(random_params)
        return checkpointer.restore(
            path.absolute(), item=random_params, restore_args=restore_args
        )

    def save(self, path: Path, state: ActorCriticState):
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path.absolute(), state.training_state)
