from typing import NamedTuple
from jax import numpy as jnp, Array
from jaxtyping import Bool, Float32

# This file is for recording the cumulative rewards for an episode


class FinishedRewardRecorderState(NamedTuple):
    finished_episode_rewards: Array
    ongoing_episode_rewards: Array


def init(vec_num: int) -> FinishedRewardRecorderState:
    return FinishedRewardRecorderState(
        finished_episode_rewards=jnp.zeros(vec_num, dtype=jnp.float32),
        ongoing_episode_rewards=jnp.zeros(vec_num, dtype=jnp.float32),
    )


def update(
    state: FinishedRewardRecorderState,
    done: Bool[Array, "vec"],
    step_rewards: Float32[Array, "vec"],
) -> tuple[FinishedRewardRecorderState, Array]:
    ongoing_episode_rewards = state.ongoing_episode_rewards + step_rewards

    finished_episode_rewards = jnp.where(
        done, ongoing_episode_rewards, state.finished_episode_rewards
    )
    ongoing_episode_rewards = jnp.where(
        done, jnp.zeros_like(ongoing_episode_rewards), ongoing_episode_rewards
    )

    return (
        FinishedRewardRecorderState(
            finished_episode_rewards=finished_episode_rewards,
            ongoing_episode_rewards=ongoing_episode_rewards,
        ),
        finished_episode_rewards,
    )
