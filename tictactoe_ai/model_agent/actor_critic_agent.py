from pathlib import Path
from typing import NamedTuple

import orbax.checkpoint as ocp
from jax import random, numpy as jnp
from jaxtyping import Int8, Float32, Key, PRNGKeyArray, Array, Bool
from flax import linen as nn

from .observation import (
    get_observation,
    get_observation_vec,
    get_available_actions,
    get_available_actions_vec,
)
from .reward import get_done, get_reward
from ..agent import Agent
from ..gamerules import GameState
from ..metrics import Metrics
from ..model.actor_critic import TrainingState
from ..model.initalize import create_actor_critic
from ..model.agent_settings import AgentSettings


class ActorCriticState(NamedTuple):
    training_state: TrainingState
    importance: Float32[Array, "vec"]


class ActorCriticAgent(Agent[ActorCriticState]):
    def __init__(self, settings: AgentSettings, total_steps: int):
        self.model = create_actor_critic(settings, total_steps)

    def initialize(self, rng_key: PRNGKeyArray, env_num: int) -> ActorCriticState:
        training_state = self.model.init(rng_key)
        return ActorCriticState(
            training_state=training_state,
            importance=jnp.ones(env_num, dtype=jnp.float32),
        )

    def act(
        self,
        agent_state: ActorCriticState,
        game_state: GameState,
        rng_key: Key[Array, ""],
    ) -> tuple[Int8[Array, ""], Float32[Array, "9"]]:
        obs = get_observation(game_state, game_state.active_player)
        available_actions = get_available_actions(game_state)
        action, logits = self.model.act(
            agent_state.training_state, obs, available_actions, rng_key
        )
        probs = nn.softmax(logits)
        return action, probs

    def learn(
        self,
        agent_state: ActorCriticState,
        game: GameState,
        actions: Int8[Array, "vec"],
        next_game: GameState,
        is_active_agent: Bool[Array, "vec"],
    ) -> tuple[ActorCriticState, Metrics]:
        player_id = jnp.where(is_active_agent, game.active_player, -game.active_player)

        obs = get_observation_vec(game, player_id)
        dones = get_done(next_game)
        available_actions = get_available_actions_vec(game)

        next_obs = get_observation_vec(next_game, player_id)
        rewards = get_reward(next_game, is_active_agent)
        # jax.debug.breakpoint()

        model_state, metrics, importance = self.model.train_step(
            agent_state.training_state,
            obs,
            available_actions,
            actions,
            rewards,
            next_obs,
            dones,
            agent_state.importance,
            is_active_agent,
        )

        return ActorCriticState(model_state, importance), metrics

    def load(self, path: Path) -> ActorCriticState:
        # random_params = self.model.init(random.PRNGKey(0))

        checkpointer = ocp.StandardCheckpointer()
        # restore_args = ocp.checkpoint_utils.construct_restore_args(random_params)
        training_state = checkpointer.restore(path.absolute())

        return ActorCriticState(training_state, jnp.ones(1, jnp.float32))

    def save(self, path: Path, state: ActorCriticState):
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path.absolute(), state.training_state)

    def get_name(self) -> str:
        return "Actor Critic"
