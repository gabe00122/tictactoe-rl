from typing import NamedTuple
from jax import numpy as jnp, Array
from jaxtyping import Array, Bool, Scalar, Int32, Float32
from . import finished_reward_recorder
from .finished_reward_recorder import (
    FinishedRewardRecorderState,
)
from .types import Metrics


class MetricsRecorderState(NamedTuple):
    finished_reward_recorder_state: FinishedRewardRecorderState
    step: Int32[Scalar, ""]
    mean_rewards: Float32[Array, "capacity"]
    state_value: Float32[Array, "capacity"]
    td_error: Float32[Array, "capacity"]
    actor_loss: Float32[Array, "capacity"]
    critic_loss: Float32[Array, "capacity"]
    entropy: Float32[Array, "capacity"]
    game_outcomes: Int32[
        Array, "capacity 5"
    ]  # agent a wins x, agent a wins o, tie, agent b wins x, agent b wins 0


def init(capacity: int, vec_num: int) -> MetricsRecorderState:
    return MetricsRecorderState(
        step=jnp.int32(0),
        finished_reward_recorder_state=finished_reward_recorder.init(vec_num),
        mean_rewards=jnp.zeros(capacity, dtype=jnp.float32),
        state_value=jnp.zeros(capacity, dtype=jnp.float32),
        td_error=jnp.zeros(capacity, dtype=jnp.float32),
        actor_loss=jnp.zeros(capacity, dtype=jnp.float32),
        critic_loss=jnp.zeros(capacity, dtype=jnp.float32),
        entropy=jnp.zeros(capacity, dtype=jnp.float32),
        game_outcomes=jnp.zeros((capacity, 5), dtype=jnp.int32),
    )


def update(
    state: MetricsRecorderState,
    # done: Bool[Array, "vec"],
    # step_rewards: Float32[Array, "vec"],
    metrics: Metrics,
) -> MetricsRecorderState:
    step = state.step
    # finished_reward_recorder_state = state.finished_reward_recorder_state
    # mean_rewards = state.mean_rewards

    # finished_reward_recorder_state, finished_rewards = finished_reward_recorder.update(
    #     finished_reward_recorder_state, done, step_rewards
    # )

    # mean_rewards = mean_rewards.at[step].set(finished_rewards.mean())

    return state._replace(
        # finished_reward_recorder_state=finished_reward_recorder_state,
        # mean_rewards=mean_rewards,
        state_value=state.state_value.at[step].set(metrics["state_value"]),
        td_error=state.td_error.at[step].set(metrics["td_error"]),
        actor_loss=state.actor_loss.at[step].set(metrics["actor_loss"]),
        critic_loss=state.critic_loss.at[step].set(metrics["critic_loss"]),
        entropy=state.entropy.at[step].set(metrics["entropy"]),
    )


def reset(state: MetricsRecorderState) -> MetricsRecorderState:
    return state._replace(
        step=jnp.int32(0),
    )
