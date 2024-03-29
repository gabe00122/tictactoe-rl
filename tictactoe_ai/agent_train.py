from functools import partial
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp, random
from jaxtyping import Array, Scalar, PRNGKeyArray, Key, Int8, Int32, Bool

from tictactoe_ai.agent import Agent
from tictactoe_ai.gamerules import (
    turn,
    reset_if_done,
    GameState,
    DRAW,
    WON,
)
from tictactoe_ai.metrics import metrics_recorder, MetricsRecorderState
from tictactoe_ai.model_agent.reward import get_done
from tictactoe_ai.util import split_n


class StaticState(NamedTuple):
    env_num: int
    opponent: Agent
    agent: Agent
    is_self_play: bool
    is_training: bool


class StepState(NamedTuple):
    rng_key: PRNGKeyArray
    opponent_state: Any
    agent_state: Any
    active_agent: Int8[Array, "envs"]
    env_state: GameState
    metrics_state: MetricsRecorderState
    step: Int32[Scalar, ""]
    total_steps: Int32[Scalar, ""]


def train_step(static_state: StaticState, step_state: StepState) -> StepState:
    env_num = static_state.env_num
    opponent = static_state.opponent
    agent = static_state.agent
    is_self_play = static_state.is_self_play
    is_training = static_state.is_training

    rng_key = step_state.rng_key
    opponent_state = step_state.opponent_state
    agent_state = step_state.agent_state
    active_agent = step_state.active_agent
    env_state = step_state.env_state
    metrics_state = step_state.metrics_state

    if is_self_play:
        opponent_state = agent_state

    rng_key, action_keys = split_n(rng_key, env_num)
    active_player = env_state.active_player

    # first_env_state: is a temporary solution
    env_state, action, first_env_state = advance_turn(
        env_state,
        active_agent,
        agent,
        agent_state,
        opponent,
        opponent_state,
        action_keys,
    )

    if is_training:
        agent_state, metrics = agent.learn(
            agent_state, first_env_state, action, env_state, active_agent == 1, step_state.step, step_state.total_steps
        )
        # opponent_state, _ = opponent.learn(
        #     opponent_state, first_env_state, action, env_state, active_agent == -1
        # )
        metrics_state = metrics_recorder.update(metrics_state, metrics)

    # record the win
    game_outcomes = get_game_outcomes(
        active_agent, active_player, env_state.over_result
    ).sum(0)
    metrics_state = record_outcome(metrics_state, game_outcomes)

    metrics_state = metrics_state._replace(step=metrics_state.step + 1)

    dones = get_done(env_state)
    rng_key, active_agent_keys = random.split(rng_key)
    active_agent = update_active_agent(active_agent, dones, active_agent_keys)

    return StepState(
        rng_key=rng_key,
        agent_state=agent_state,
        opponent_state=opponent_state,
        env_state=env_state,
        active_agent=active_agent,
        metrics_state=metrics_state,
        step=step_state.step + 1,
        total_steps=step_state.total_steps,
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

    action, _ = jax.lax.cond(
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
            step - jit_iterations: step
        ]

        total_games = step_state.metrics_state.game_outcomes.sum().item()
        epoch_games = game_outcomes.sum().item()
        agent_a_x, agent_a_o, ties, agent_b_x, agent_b_o = (game_outcomes.sum(0) / epoch_games).tolist()

        agent_a_name = static_state.opponent.get_name()
        agent_b_name = static_state.agent.get_name()

        if static_state.is_self_play:
            agent_a_x += agent_b_x
            agent_a_o += agent_b_o

        print(
            f"step: {(i+1) * jit_iterations}, total games: {total_games}, total steps: {(i+1) * jit_iterations * static_state.env_num}"
        )
        print(f"  {agent_a_name} x: {agent_a_x:.0%}")
        print(f"  {agent_a_name} o: {agent_a_o:.0%}")
        print(f"  Ties: {ties:.0%}")
        if not static_state.is_self_play:
            print(f"  {agent_b_name} x: {agent_b_x:.0%}")
            print(f"  {agent_b_name} o: {agent_b_o:.0%}")
        print()
    return step_state
