import shutil
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray, Array, Float
from orbax.checkpoint import PyTreeCheckpointer

from tictactoe_ai.gamerules.initialize import initialize_n_games
from tictactoe_ai.gamerules.turn import turn, reset_if_done
from tictactoe_ai.gamerules.types import VectorizedGameState
from tictactoe_ai.metrics import metrics_recorder, MetricsRecorderState
from tictactoe_ai.metrics.metrics_logger_np import MetricsLoggerNP
from tictactoe_ai.minmax.minmax_player import get_action
from tictactoe_ai.model.actor_critic import ActorCritic
from tictactoe_ai.model.actor_critic import TrainingState
from tictactoe_ai.model.initalize import create_actor_critic
from tictactoe_ai.model.run_settings import RunSettings
from tictactoe_ai.model.run_settings import save_settings
from tictactoe_ai.observation import get_available_actions_vec, get_observation_vec
from tictactoe_ai.reward import get_reward, get_done
from tictactoe_ai.util import split_n
from tictactoe_ai.agent import Agent


class StaticState(NamedTuple):
    env_num: int
    agent_a: Agent
    agent_b: Agent


class StepState(NamedTuple):
    rng_key: PRNGKeyArray
    agent_a_state: Any
    agent_b_state: Any
    env_state: VectorizedGameState
    metrics_state: MetricsRecorderState


def train_step(static_state: StaticState, step_state: StepState) -> StepState:
    env_num = static_state.env_num
    actor_critic = static_state.actor_critic

    rng_key = step_state.rng_key
    training_state = step_state.training_state
    env_state = step_state.env_state
    importance = step_state.importance
    metrics_state = step_state.metrics_state

    # reset finished games
    env_state = jax.vmap(reset_if_done)(env_state)

    # pick an action
    before_obs = get_observation_vec(env_state, 1)
    available_actions = get_available_actions_vec(env_state)

    act_vec = jax.vmap(actor_critic.act, (None, 0, 0, 0))

    rng_key, action_keys = split_n(rng_key, env_num)
    actions = act_vec(training_state, before_obs, available_actions, action_keys)

    # play the action
    turn_vec = jax.vmap(turn, (0, 0))
    env_state = turn_vec(env_state, actions)

    # gather the after state info to learn from later
    after_obs = get_observation_vec(env_state, 1)
    reward = get_reward(env_state, 1)
    done = get_done(env_state)

    training_state, metrics, importance = actor_critic.train_step(
        training_state,
        before_obs,
        available_actions,
        actions,
        reward,
        after_obs,
        done,
        importance,
        True,
    )

    # log training metrics
    metrics_state = metrics_recorder.update(metrics_state, done, reward, metrics)

    # reset
    env_state = jax.vmap(reset_if_done)(env_state)

    # re-poll this in case of game over
    after_obs = get_observation_vec(env_state, 1)

    # play a move as a random opponent
    rng_key, action_keys = split_n(rng_key, env_num)

    # opponent_obs = get_observation_vec(env_state, -1)
    # available_actions = get_available_actions_vec(env_state)
    # opponent_actions = act_vec(training_state, opponent_obs, available_actions, action_keys)
    get_random_move_vec = jax.vmap(get_action, (None, 0, 0))
    opponent_actions = get_random_move_vec(optimal_play, env_state, action_keys)
    env_state = turn_vec(env_state, opponent_actions)

    # train the critic from the other state transition
    opponent_after_state = get_observation_vec(env_state, 1)
    reward = get_reward(env_state, 1)
    done = get_done(env_state)

    training_state, metrics, importance = actor_critic.train_step(
        training_state,
        after_obs,
        available_actions,  # ignored
        actions,  # ignored
        reward,
        opponent_after_state,
        done,
        importance,
        False,
    )

    # log training metrics
    metrics_state = metrics_recorder.update(metrics_state, done, reward, metrics)

    return StepState(
        env_state=env_state,
        importance=importance,
        rng_key=rng_key,
        training_state=training_state,
        metrics_state=metrics_state,
    )


@partial(jax.jit, static_argnums=(0, 1), donate_argnums=(2,))
def jit_train_n_steps(
    static_state: StaticState, iterations: int, step_state: StepState
) -> StepState:
    return jax.lax.fori_loop(
        0, iterations, lambda _, step: train_step(static_state, step), step_state
    )


def train_n_steps(
    static_state: StaticState,
    total_iterations: int,
    jit_iterations: int,
    step_state: StepState,
) -> StepState:
    for i in range(total_iterations // jit_iterations):
        step_state = jit_train_n_steps(static_state, jit_iterations, step_state)

        step = step_state.metrics_state.step
        rewards = step_state.metrics_state.mean_rewards[step - jit_iterations : step]
        print(f"step: {i * jit_iterations}, reward: {rewards.mean().item()}")
    return step_state


def main():
    settings = RunSettings(
        git_hash="blank",
        env_name="tictactoe",
        seed=33,
        total_steps=50_000,
        env_num=64,
        discount=0.99,
        root_hidden_layers=[64, 64],
        actor_hidden_layers=[64],
        critic_hidden_layers=[64],
        actor_last_layer_scale=0.01,
        critic_last_layer_scale=1.0,
        learning_rate=0.001,
        actor_coef=0.15,
        critic_coef=1.0,
        optimizer="adamw",
        adam_beta=0.997,
        weight_decay=0.001,
    )

    rng_key = random.PRNGKey(settings["seed"])
    actor_critic = create_actor_critic(settings)

    rng_key, model_key = random.split(rng_key)
    model_training_state = actor_critic.init(model_key)

    total_steps = settings["total_steps"]
    jit_iterations = 1_000
    env_num = settings["env_num"]

    static_state = StaticState(
        env_num=settings["env_num"],
        actor_critic=actor_critic,
    )

    game_state = initialize_n_games(env_num)
    step_state = StepState(
        rng_key=rng_key,
        env_state=game_state,
        importance=jnp.ones(env_num, dtype=jnp.float32),
        training_state=model_training_state,
        metrics_state=metrics_recorder.init(total_steps * 2, env_num),
    )
    step_state = train_n_steps(static_state, total_steps, jit_iterations, step_state)

    metrics_logger = MetricsLoggerNP(total_steps * 2)
    metrics_logger.log(step_state.metrics_state)

    save_path = Path("./run-selfplay")
    create_directory(save_path)
    save_settings(save_path / "settings.json", settings)
    save_params(save_path / "model", step_state.training_state.model_params)
    save_metrics(save_path / "metrics.parquet", metrics_logger)


def create_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_params(path: Path, params: Any):
    checkpointer = PyTreeCheckpointer()
    checkpointer.save(path.absolute(), params)


def save_metrics(path: Path, metrics: MetricsLoggerNP):
    metrics.get_dataframe().to_parquet(path, index=False)


if __name__ == "__main__":
    main()
