from typing import NamedTuple
from jax import numpy as jnp, Array
from . import finished_reward_recorder
from .finished_reward_recorder import (
    FinishedRewardRecorderState,
)
from .type import Metrics


class MetricsRecorderState(NamedTuple):
    step: Array
    mean_rewards: Array
    finished_reward_recorder_state: FinishedRewardRecorderState
    state_value: Array
    td_error: Array
    actor_loss: Array
    critic_loss: Array
    entropy: Array

    winsX: Array
    winsO: Array
    ties: Array


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
        winsX=jnp.int32(0),
        winsO=jnp.int32(0),
        ties=jnp.int32(0),
    )


def update(
    state: MetricsRecorderState,
    done: Array,
    step_rewards: Array,
    metrics: Metrics,
) -> MetricsRecorderState:
    step = state.step
    finished_reward_recorder_state = state.finished_reward_recorder_state
    mean_rewards = state.mean_rewards

    finished_reward_recorder_state, finished_rewards = (
        finished_reward_recorder.update(
            finished_reward_recorder_state, done, step_rewards
        )
    )

    mean_rewards = mean_rewards.at[step].set(finished_rewards.mean())
    step = step + 1

    return state._replace(
        step=step,
        finished_reward_recorder_state=finished_reward_recorder_state,
        mean_rewards=mean_rewards,
        state_value=state.state_value.at[step].set(metrics["state_value"]),
        td_error=state.td_error.at[step].set(metrics["td_error"]),
        actor_loss=state.actor_loss.at[step].set(metrics["actor_loss"]),
        critic_loss=state.critic_loss.at[step].set(metrics["critic_loss"]),
        entropy=state.entropy.at[step].set(metrics["entropy"]),
    )


def reset(state: MetricsRecorderState) -> MetricsRecorderState:
    return state._replace(
        step=jnp.int32(0),
        winsA=jnp.int32(0),
        winsB=jnp.int32(0),
        ties=jnp.int32(0),
    )
