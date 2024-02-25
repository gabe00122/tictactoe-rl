from dataclasses import dataclass, field
from typing import NamedTuple
from jax import numpy as jnp, Array
from .finished_reward_recorder import FinishedRewardRecorder, FinishedRewardRecorderState


class MetricsRecorderState(NamedTuple):
    step: Array
    mean_rewards: Array
    finished_reward_recorder_state: FinishedRewardRecorderState


@dataclass(frozen=True)
class MetricsRecorder:
    step_num: int = field()
    finished_reward_recorder: FinishedRewardRecorder = field()

    @classmethod
    def create(cls, step_num: int, vec_num: int) -> "MetricsRecorder":
        return cls(
            step_num=step_num,
            finished_reward_recorder=FinishedRewardRecorder(vec_num),
        )

    def init(self) -> MetricsRecorderState:
        return MetricsRecorderState(
            step=jnp.int32(0),
            finished_reward_recorder_state=self.finished_reward_recorder.init(),
            mean_rewards=jnp.zeros((self.step_num,), dtype=jnp.float32),
        )

    def update(self, state: MetricsRecorderState, done: Array, step_rewards: Array) -> MetricsRecorderState:
        step = state.step
        finished_reward_recorder_state = state.finished_reward_recorder_state
        mean_rewards = state.mean_rewards

        finished_reward_recorder_state, finished_rewards = self.finished_reward_recorder.update(
            finished_reward_recorder_state, done, step_rewards
        )

        mean_rewards = mean_rewards.at[step].set(finished_rewards.mean())
        step = step + 1

        return MetricsRecorderState(
            step=step,
            finished_reward_recorder_state=finished_reward_recorder_state,
            mean_rewards=mean_rewards,
        )

    def reset(self, state: MetricsRecorderState) -> MetricsRecorderState:
        return MetricsRecorderState(
            finished_reward_recorder_state=state.finished_reward_recorder_state,
            step=jnp.int32(0),
            mean_rewards=jnp.zeros((self.step_num,), dtype=jnp.float32),
        )
