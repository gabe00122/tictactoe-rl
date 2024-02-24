import jax
from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray, Array, Scalar, Float, Bool
from typing import TypedDict, Any, NamedTuple
from functools import partial
from tictactoe_ai.model.actor_critic import TrainingState
from tictactoe_ai.model.initalize import create_actor_critic
from tictactoe_ai.model.run_settings import RunSettings
from tictactoe_ai.gamerules.initialize import initialize_n_games
from tictactoe_ai.gamerules.turn import turn
from tictactoe_ai.gamerules.types import GameState
from tictactoe_ai.model.actor_critic import ActorCritic
from tictactoe_ai.observation import get_available_actions, get_beforestate_observation, get_afterstate_observation


class StaticState(NamedTuple):
    env_num: int
    actor_critic: ActorCritic


class StepState(TypedDict):
    rng_key: PRNGKeyArray
    training_state: TrainingState
    env_state: Any  # vectorized GameState
    importance: Float[Array, "vec"]


def train_step(static_state: StaticState, step_state: StepState) -> StepState:
    env_num = static_state.env_num
    actor_critic = static_state.actor_critic

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
        lambda: jnp.int8(0)
    )


def get_done(state: GameState) -> Bool[Scalar, ""]:
    return state["over_result"]["is_over"]


def split_n(rng_key: PRNGKeyArray, num: int) -> tuple[PRNGKeyArray, PRNGKeyArray]:
    keys = random.split(rng_key, num + 1)
    return keys[0], keys[1:]


@partial(jax.jit, static_argnums=(0, 1), donate_argnums=(2,))
def jit_train_n_steps(static_state: StaticState, iterations: int, step_state: StepState) -> StepState:
    return jax.lax.fori_loop(
        0,
        iterations,
        lambda _, step: train_step(static_state, step),
        step_state
    )


def train_n_steps(static_state: StaticState, total_iterations: int, jit_iterations: int, step_state: StepState) -> StepState:
    for i in range(total_iterations // jit_iterations):
        step_state = jit_train_n_steps(static_state, jit_iterations, step_state)
        print(f"step: {i * jit_iterations}")
    return step_state


def main():
    settings = RunSettings(
        git_hash='blank',
        env_name='tictactoe',
        seed=4321,
        total_steps=100_000,
        env_num=8,
        discount=0.99,
        root_hidden_layers=[64],
        actor_hidden_layers=[64],
        critic_hidden_layers=[64],
        actor_last_layer_scale=0.01,
        critic_last_layer_scale=1.0,
        learning_rate=0.0001,
        actor_coef=0.25,
        critic_coef=1.0,
        optimizer='adamw',
        adam_beta=0.97,
        weight_decay=0.0,
    )

    rng_key = random.PRNGKey(settings['seed'])
    actor_critic = create_actor_critic(settings)

    rng_key, model_key = random.split(rng_key)
    model_training_state = actor_critic.init(model_key)

    static_state = StaticState(
        env_num=settings['env_num'],
        actor_critic=actor_critic,
    )

    game_state = initialize_n_games(settings['env_num'])
    step_state: StepState = {
        'rng_key': rng_key,
        'env_state': game_state,
        'importance': jnp.ones((settings['env_num']), dtype=jnp.float32),
        'training_state': model_training_state
    }
    step_state = train_n_steps(static_state, settings['total_steps'], 1_000, step_state)


if __name__ == '__main__':
    main()
