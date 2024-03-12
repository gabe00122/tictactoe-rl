import shutil
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp, random
from jaxtyping import PRNGKeyArray
from orbax.checkpoint import PyTreeCheckpointer

from tictactoe_ai.agent import Agent
from tictactoe_ai.gamerules.initialize import initialize_n_games
from tictactoe_ai.gamerules.turn import turn, reset_if_done
from tictactoe_ai.gamerules.types import VectorizedGameState
from tictactoe_ai.metrics import metrics_recorder, MetricsRecorderState
from tictactoe_ai.metrics.metrics_logger_np import MetricsLoggerNP
from tictactoe_ai.model.run_settings import save_settings
from tictactoe_ai.util import split_n
from tictactoe_ai.random_agent import RandomAgent


class StaticState(NamedTuple):
    env_num: int
    agent_a: Agent
    agent_b: Agent


class StepState(NamedTuple):
    rng_key: PRNGKeyArray
    state_a: Any
    state_b: Any
    env_state: VectorizedGameState
    metrics_state: MetricsRecorderState
    game_outcomes: Any


def train_step(static_state: StaticState, step_state: StepState) -> StepState:
    env_num = static_state.env_num
    agent_a = static_state.agent_a
    agent_b = static_state.agent_b

    rng_key = step_state.rng_key
    state_a = step_state.state_a
    state_b = step_state.state_b
    env_state = step_state.env_state
    metrics_state = step_state.metrics_state
    game_outcomes = step_state.game_outcomes

    def get_actions(a, b, state, a_keys):
        actions_a = agent_a.act(a, state, a_keys)
        actions_b = agent_b.act(b, state, a_keys)
        return jnp.where(state.active_player, actions_a, actions_b)

    # reset finished games
    rng_key, initialize_keys = split_n(rng_key, env_num)
    env_state = jax.vmap(reset_if_done)(env_state, initialize_keys)

    # pick an action
    rng_key, action_keys = split_n(rng_key, env_num)
    actions = get_actions(state_a, state_b, env_state, action_keys)

    # play the action
    env_state = jax.vmap(turn, (0, 0))(env_state, actions)

    game_outcomes += jax.nn.one_hot(env_state.over_result - 1, 3, dtype=jnp.int32).sum(0)

    return StepState(
        rng_key=rng_key,
        state_a=state_a,
        state_b=state_b,
        env_state=env_state,
        metrics_state=metrics_state,
        game_outcomes=game_outcomes,
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
    rng_key = random.PRNGKey(123)

    total_steps = 5000
    jit_iterations = 1_000
    env_num = 128

    static_state = StaticState(
        env_num=env_num,
        agent_a=RandomAgent(),
        agent_b=RandomAgent(),
    )

    rng_key, game_keys = split_n(rng_key, env_num)
    game_state = initialize_n_games(game_keys)

    rng_key, agent_a_key, agent_b_key = random.split(rng_key, 3)
    step_state = StepState(
        rng_key=rng_key,
        state_a=static_state.agent_a.initialize(agent_a_key),
        state_b=static_state.agent_b.initialize(agent_b_key),
        env_state=game_state,
        metrics_state=metrics_recorder.init(total_steps * 2, env_num),
        game_outcomes=jnp.zeros(3, dtype=jnp.int32)
    )
    step_state = train_n_steps(static_state, total_steps, jit_iterations, step_state)

    print(step_state.game_outcomes)

    metrics_logger = MetricsLoggerNP(total_steps * 2)
    metrics_logger.log(step_state.metrics_state)

    save_path = Path("./run-selfplay")
    create_directory(save_path)
    # save_settings(save_path / "settings.json", settings)
    # save_params(save_path / "model", step_state.training_state.model_params)
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
