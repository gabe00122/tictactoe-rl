from dataclasses import dataclass, field
from typing import NamedTuple
from jax import numpy as jnp, Array


class FinishedRewardRecorderState(NamedTuple):
    finished_episode_rewards: Array
    ongoing_episode_rewards: Array


@dataclass(frozen=True)
class FinishedRewardRecorder:
    vec_num: int = field()

    def init(self) -> FinishedRewardRecorderState:
        return FinishedRewardRecorderState(
            finished_episode_rewards=jnp.zeros((self.vec_num,), dtype=jnp.float32),
            ongoing_episode_rewards=jnp.zeros((self.vec_num,), dtype=jnp.float32),
        )

    def update(
        self, state: FinishedRewardRecorderState, done: Array, step_rewards: Array
    ) -> tuple[FinishedRewardRecorderState, Array]:
        ongoing_episode_rewards = state.ongoing_episode_rewards + step_rewards

        finished_episode_rewards = jnp.where(
            done, ongoing_episode_rewards, state.finished_episode_rewards
        )
        ongoing_episode_rewards = jnp.where(
            done, jnp.zeros((self.vec_num,)), ongoing_episode_rewards
        )

        return (
            FinishedRewardRecorderState(
                finished_episode_rewards=finished_episode_rewards,
                ongoing_episode_rewards=ongoing_episode_rewards,
            ),
            finished_episode_rewards,
        )
