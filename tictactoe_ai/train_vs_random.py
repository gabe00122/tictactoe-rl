import jax
from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray, Array, Scalar, Float, Int, Bool
from typing import TypedDict, Any
from functools import partial
from .model.actor_critic import TrainingState
from .gamerules.turn import turn
from .gamerules.types import GameState
from .model.actor_critic import ActorCritic
from .observation import get_available_actions, get_beforestate_observation, get_afterstate_observation


class StaticState(TypedDict):
    env_num: int
    iterations: int
    actor_critic: ActorCritic


class StepState(TypedDict):
    rng_key: PRNGKeyArray
    training_state: TrainingState
    env_state: Any  # vectorized GameState
    importance: Float[Array, "vec"]


def train_step(static_state: StaticState, step_state: StepState) -> StepState:
    env_num = static_state["env_num"]
    actor_critic = static_state["actor_critic"]

    rng_key = step_state["rng_key"]
    training_state = step_state["training_state"]
    env_state = step_state["env_state"]
    importance = step_state["importance"]

    # pick an action
    before_obs = jax.vmap(get_beforestate_observation)(env_state)

    rng_key, action_keys = split_n(rng_key, env_num)
    available_actions = jax.vmap(get_available_actions)(env_state)
    v_act = jax.vmap(actor_critic.act, (None, 0, 0, 0))
    actions = v_act(training_state, before_obs, available_actions, action_keys)

    # play the action
    v_turn = jax.vmap(turn, (0, 0))
    env_state = v_turn(env_state, actions)
    after_obs = jax.vmap(get_afterstate_observation)(env_state)

    # learn from the action
    reward = jax.vmap(get_reward)(env_state)
    done = jax.vmap(get_done)(env_state)
    training_state, metrics, importance = actor_critic.train_step(training_state, before_obs, available_actions, actions, reward, after_obs, done, importance)

    # play the opponent action, no training is happening from this action for now
    opponent_obs = jax.vmap(get_beforestate_observation)(env_state)
    rng_key, action_keys = split_n(rng_key, env_num)
    available_actions = jax.vmap(get_available_actions)(env_state)
    actions = v_act(training_state, opponent_obs, available_actions, action_keys)
    env_state = v_turn(env_state, actions)

    return {
        'env_state': env_state,
        'importance': importance,
        'rng_key': rng_key,
        'training_state': training_state,
    }


def get_reward(state: GameState) -> Float[Scalar, ""]:
    result = state['over_result']
    is_over = result['is_over']
    winner = result['winner']
    previous_active_player = -state['active_player']

    return jax.lax.cond(
        is_over,
        lambda: jax.lax.cond(previous_active_player == 1, lambda: winner, lambda: -winner),
        lambda: 0
    )


def get_done(state: GameState) -> Bool[Scalar, ""]:
    return state["over_result"]["is_over"]


def split_n(rng_key: PRNGKeyArray, num: int) -> tuple[PRNGKeyArray, PRNGKeyArray]:
    keys = random.split(rng_key, num + 1)
    return keys[0], keys[1:]


@partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
def train_n_steps(static_state: StaticState, step_state: StepState) -> StepState:
    return jax.lax.fori_loop(
        0,
        static_state["iterations"],
        lambda _, step: train_step(static_state, step),
        step_state
    )
