import shutil
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp, random
from jaxtyping import Array, Scalar, PRNGKeyArray, Key, Int8, Int32, Bool
from orbax.checkpoint import PyTreeCheckpointer

from tictactoe_ai.agent import Agent
from tictactoe_ai.gamerules.initialize import initialize_n_games
from tictactoe_ai.gamerules.turn import turn, reset_if_done
from tictactoe_ai.gamerules.types import GameState
from tictactoe_ai.gamerules.over import DRAW, ONGOING, WON
from tictactoe_ai.metrics import metrics_recorder, MetricsRecorderState
from tictactoe_ai.metrics.metrics_logger_np import MetricsLoggerNP
from tictactoe_ai.model.run_settings import save_settings
from tictactoe_ai.reward import get_done
from tictactoe_ai.util import split_n
from tictactoe_ai.random_agent import RandomAgent
from tictactoe_ai.minmax.minmax_player import MinmaxAgent


class StaticState(NamedTuple):
    env_num: int
    agent_a: Agent
    agent_b: Agent


class StepState(NamedTuple):
    rng_key: PRNGKeyArray
    state_a: Any
    state_b: Any
    active_agent: Int8[Array, "envs"]
    env_state: GameState
    metrics_state: MetricsRecorderState


def train_step(static_state: StaticState, step_state: StepState) -> StepState:
    env_num, agent_a, agent_b = static_state
    rng_key, state_a, state_b, active_agent, env_state, metrics_state = step_state


    rng_key, action_keys = split_n(rng_key, env_num)
    env_state = advance_turn(env_state, active_agent, agent_a, state_a, agent_b, state_b, action_keys)

    # record the win
    game_outcomes = get_game_outcomes(active_agent, env_state.over_result).sum(0)
    metrics_state = record_outcome(metrics_state, game_outcomes)

    dones = get_done(env_state)
    rng_key, active_agent_keys = random.split(rng_key)
    active_agent = update_active_agent(active_agent, dones, active_agent_keys)

    return StepState(
        rng_key=rng_key,
        state_a=state_a,
        state_b=state_b,
        env_state=env_state,
        active_agent=active_agent,
        metrics_state=metrics_state,
    )

@partial(jax.vmap, in_axes=(0, 0, None, None, None, None, 0))
def advance_turn(
        env_state: GameState,
        active_agent: Int8[Scalar, ""],
        agent_a: Agent,
        state_a: Any,
        agent_b: Agent,
        state_b: Any,
        rng_key: PRNGKeyArray
) -> GameState:
    env_state = reset_if_done(env_state)

    action = jax.lax.cond(
        active_agent == 1,
        lambda: agent_a.act(state_a, env_state, rng_key),
        lambda: agent_b.act(state_b, env_state, rng_key),
    )

    env_state = turn(env_state, action)
    return env_state


@partial(jax.vmap, in_axes=(0, 0))
def get_game_outcomes(active_agent, over_result):
    return jnp.array(
        [
            jnp.logical_and(active_agent == -1, over_result == WON),
            over_result == DRAW,
            jnp.logical_and(active_agent == 1, over_result == WON),
        ],
        dtype=jnp.int32,
    )


def record_outcome(metrics: MetricsRecorderState, game_outcomes: Int32[Array, "3"]) -> MetricsRecorderState:
    return metrics._replace(
        step=metrics.step + 1,
        game_outcomes=metrics.game_outcomes.at[metrics.step].set(game_outcomes),
    )


def update_active_agent(active_agent: Int8[Array, "envs"], done: Bool[Array, "envs"], rng_key: Key[Scalar, ""]) -> Int8[Array, "envs"]:
    shape = active_agent.shape
    random_active_agents = random.choice(rng_key, jnp.array([-1, 1], dtype=jnp.int8), shape)
    return jnp.where(done, random_active_agents, active_agent)


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
        # rewards = step_state.metrics_state.mean_rewards[step - jit_iterations : step]
        game_outcomes = step_state.metrics_state.game_outcomes[
            step - jit_iterations : step
        ]
        agent_b, ties, agent_a = game_outcomes.sum(0).tolist()
        print(
            f"step: {i * jit_iterations}, agent a: {agent_a}, ties: {ties}, agent b: {agent_b}"
        )
    return step_state


def main():
    rng_key = random.PRNGKey(123)

    total_steps = 5000
    jit_iterations = 1_000
    env_num = 128

    static_state = StaticState(
        env_num=env_num,
        agent_a=RandomAgent(),
        agent_b=MinmaxAgent(),
    )

    game_state = initialize_n_games(env_num)

    rng_key, agent_a_key, agent_b_key, active_agent_keys = random.split(rng_key, 4)
    active_agents = random.choice(
        active_agent_keys, jnp.array([-1, 1], dtype=jnp.int8), (env_num,)
    )

    step_state = StepState(
        rng_key=rng_key,
        state_a=static_state.agent_a.initialize(agent_a_key),
        state_b=static_state.agent_b.load(Path("./optimal_play.npy")),
        active_agent=active_agents,
        env_state=game_state,
        metrics_state=metrics_recorder.init(total_steps, env_num),
    )
    step_state = train_n_steps(static_state, total_steps, jit_iterations, step_state)

    metrics_logger = MetricsLoggerNP(total_steps)
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
