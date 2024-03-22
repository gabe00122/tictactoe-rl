import shutil
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp, random
from jaxtyping import Array, Scalar, PRNGKeyArray, Key, Int8, Int32, Bool

from tictactoe_ai.agent import Agent
from tictactoe_ai.gamerules import (
    initialize_n_games,
    turn,
    reset_if_done,
    GameState,
    DRAW,
    ONGOING,
    WON,
)
from tictactoe_ai.metrics import metrics_recorder, MetricsRecorderState
from tictactoe_ai.metrics.metrics_logger_np import MetricsLoggerNP
from tictactoe_ai.model_agent import ActorCriticAgent
from tictactoe_ai.model_agent.reward import get_done
from tictactoe_ai.util import split_n
from tictactoe_ai.random_agent import RandomAgent
from tictactoe_ai.minmax.minmax_player import MinmaxAgent
from tictactoe_ai.model.run_settings import load_settings
from tictactoe_ai.model.initalize import create_actor_critic


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
    active_player = env_state.active_player

    # first_env_state: is a temporary solution
    env_state, action, first_env_state = advance_turn(
        env_state, active_agent, agent_a, state_b, agent_b, state_b, action_keys
    )

    # state_a, _ = static_state.agent_a.learn(state_a, first_env_state, action, env_state, active_agent == 1)
    state_b, metrics = static_state.agent_b.learn(
        state_b, first_env_state, action, env_state, active_agent == -1
    )

    # record the win
    game_outcomes = get_game_outcomes(
        active_agent, active_player, env_state.over_result
    ).sum(0)
    metrics_state = record_outcome(metrics_state, game_outcomes)
    metrics_state = metrics_recorder.update(metrics_state, metrics)

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
    rng_key: PRNGKeyArray,
) -> tuple[GameState, Any, GameState]:
    env_state = reset_if_done(env_state)
    first_env = env_state

    action = jax.lax.cond(
        active_agent == 1,
        lambda: agent_a.act(state_a, env_state, rng_key),
        lambda: agent_b.act(state_b, env_state, rng_key),
    )

    env_state = turn(env_state, action)
    return env_state, action, first_env


@partial(jax.vmap, in_axes=(0, 0, 0))
def get_game_outcomes(active_agent, active_player, over_result):
    return jnp.array(
        [
            jnp.logical_and(
                jnp.logical_and(active_agent == -1, over_result == WON),
                active_player == 1,
            ),
            jnp.logical_and(
                jnp.logical_and(active_agent == -1, over_result == WON),
                active_player == -1,
            ),
            over_result == DRAW,
            jnp.logical_and(
                jnp.logical_and(active_agent == 1, over_result == WON),
                active_player == 1,
            ),
            jnp.logical_and(
                jnp.logical_and(active_agent == 1, over_result == WON),
                active_player == -1,
            ),
        ],
        dtype=jnp.int32,
    )


def record_outcome(
    metrics: MetricsRecorderState, game_outcomes: Int32[Array, "5"]
) -> MetricsRecorderState:
    return metrics._replace(
        # step=metrics.step + 1,
        game_outcomes=metrics.game_outcomes.at[metrics.step].set(game_outcomes),
    )


def update_active_agent(
    active_agent: Int8[Array, "envs"],
    done: Bool[Array, "envs"],
    rng_key: Key[Scalar, ""],
) -> Int8[Array, "envs"]:
    shape = active_agent.shape
    random_active_agents = random.choice(
        rng_key, jnp.array([-1, 1], dtype=jnp.int8), shape
    )
    return jnp.where(done, random_active_agents, -active_agent)


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
        total_games = step_state.metrics_state.game_outcomes.sum().item()
        agent_b_x, agent_b_o, ties, agent_a_x, agent_a_o = game_outcomes.sum(0).tolist()

        agent_a_name = static_state.agent_a.get_name()
        agent_b_name = static_state.agent_b.get_name()

        print(
            f"step: {(i+1) * jit_iterations}, total games: {total_games}, total steps: {(i+1) * jit_iterations * static_state.env_num}"
        )
        print(f"  {agent_a_name} x: {agent_a_x}")
        print(f"  {agent_a_name} o: {agent_a_o}")
        print(f"  Ties: {ties}")
        print(f"  {agent_b_name} x: {agent_b_x}")
        print(f"  {agent_b_name} o: {agent_b_o}")
        print()
    return step_state


def main():
    rng_key = random.PRNGKey(123)

    agent_settings = load_settings("./experiments/standard.json")
    total_steps = agent_settings["total_steps"]
    env_num = agent_settings["env_num"]
    jit_iterations = 1_000

    actor_critic = create_actor_critic(agent_settings)

    static_state = StaticState(
        env_num=env_num,
        agent_a=RandomAgent(),
        agent_b=ActorCriticAgent(actor_critic),
    )

    game_state = initialize_n_games(env_num)

    rng_key, agent_a_key, agent_b_key, active_agent_keys = random.split(rng_key, 4)
    active_agents = random.choice(
        active_agent_keys, jnp.array([-1, 1], dtype=jnp.int8), (env_num,)
    )

    step_state = StepState(
        rng_key=rng_key,
        state_a=static_state.agent_a.initialize(agent_a_key, env_num),
        state_b=static_state.agent_b.initialize(agent_b_key, env_num),
        active_agent=active_agents,
        env_state=game_state,
        metrics_state=metrics_recorder.init(total_steps, env_num),
    )
    step_state = train_n_steps(static_state, total_steps, jit_iterations, step_state)

    metrics_logger = MetricsLoggerNP(total_steps)
    metrics_logger.log(step_state.metrics_state)

    save_path = Path("./run-selfplay")
    create_directory(save_path)
    save_metrics(save_path / "metrics.parquet", metrics_logger)
    static_state.agent_b.save(save_path / "checkpoint", step_state.state_b)


def create_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_metrics(path: Path, metrics: MetricsLoggerNP):
    metrics.get_dataframe().to_parquet(path, index=False)


if __name__ == "__main__":
    main()
