from typing import Any, NamedTuple
import jax
from jax import numpy as jnp, random
from flax import linen as nn, struct
from flax.core.frozen_dict import FrozenDict, Mapping
from flax.struct import PyTreeNode
import optax
from jaxtyping import Array, Scalar, Float32, Bool, PRNGKeyArray, Int
from .actor_critic_model import ActorCriticModel
from tictactoe_ai.metrics.types import Metrics
from .entropy import entropy_loss


type ModelParams = FrozenDict[str, Mapping[str, Any]] | dict[str, Any]


class TrainingState(NamedTuple):
    model_params: ModelParams
    opt_state: optax.OptState


class ActorCritic(PyTreeNode):
    model: ActorCriticModel = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    discount: float = struct.field(default=0.99)

    actor_coef: float = struct.field(default=0.5)
    entropy_coef: float = struct.field(default=0.0001)

    def __hash__(self):
        return id(self)

    def init(self, key: PRNGKeyArray) -> TrainingState:
        observation_dummy = jnp.zeros((9 * 3), jnp.float32)
        mask_dummy = jnp.full((9,), True)
        model_params = self.model.init(key, observation_dummy, mask_dummy)

        return TrainingState(
            model_params=model_params,
            opt_state=self.optimizer.init(model_params),
        )

    def act(
        self,
        params: TrainingState,
        obs: Float32[Array, "9 3"],
        avalible_actions: Bool[Array, "9"],
        key: PRNGKeyArray,
    ):
        logits, _ = self.model.apply(params.model_params, obs, avalible_actions)
        return random.categorical(key, logits), logits

    def loss(
        self,
        model_params: ModelParams,
        obs: Float32[Array, "vec 9 3"],
        available_actions: Bool[Array, "vec 9"],
        actions: Int[Array, "vec"],
        rewards: Float32[Array, "vec"],
        next_obs: Float32[Array, "vec 9 3"],
        done: Bool[Array, "vec"],
        importance: Float32[Array, "vec"],
        took_turn: Bool[Array, "vec"],
    ) -> tuple[Float32[Scalar, ""], Metrics]:
        v_model = jax.vmap(self.model.apply, (None, 0, 0), (0, 0))
        v_entropy_loss = jax.vmap(entropy_loss)
        v_log_softmax = jax.vmap(nn.log_softmax)

        action_logits, vf_values = v_model(model_params, obs, available_actions)

        _, next_vf_values = v_model(
            jax.lax.stop_gradient(model_params), next_obs, available_actions
        )

        target_value = jnp.where(
            done,
            rewards,
            rewards + self.discount * next_vf_values,
        )

        td_error = target_value - vf_values
        critic_loss = (td_error**2).mean()

        # actor loss
        action_probs = v_log_softmax(action_logits)
        selected_action_prob = action_probs[jnp.arange(action_probs.shape[0]), actions]

        actor_loss, entropy = jax.lax.cond(
            took_turn.any(),
            lambda: (
                -jnp.mean(selected_action_prob * td_error * importance, where=took_turn),
                jnp.mean(v_entropy_loss(action_probs), where=took_turn),
            ),
            lambda: (jnp.float32(0), jnp.float32(0)),
        )

        loss = self.actor_coef * actor_loss + critic_loss + self.entropy_coef * entropy

        # jax.debug.breakpoint()

        metrics: Metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "td_error": jnp.mean(td_error),
            "state_value": jnp.mean(target_value),
            "entropy": entropy,
        }

        return loss, metrics

    def update_model(
        self,
        params: TrainingState,
        obs: Float32[Array, "vec 9 3"],
        available_actions: Bool[Array, "vec 9"],
        actions: Int[Array, "vec"],
        rewards: Float32[Array, "vec"],
        next_obs: Float32[Array, "vec 9 3"],
        done: Bool[Array, "vec"],
        importance: Float32[Array, "vec"],
        took_turn: Bool[Array, "vec"],
    ):
        loss_fn = jax.value_and_grad(self.loss, has_aux=True)
        (loss, metrics), grad = loss_fn(
            params.model_params,
            obs,
            available_actions,
            actions,
            rewards,
            next_obs,
            done,
            importance,
            took_turn,
        )

        updates, opt_state = self.optimizer.update(
            grad, params.opt_state, params.model_params
        )
        model_params = optax.apply_updates(params.model_params, updates)

        params = TrainingState(model_params, opt_state)
        return params, metrics

    def train_step(
        self,
        params: TrainingState,
        obs: Float32[Array, "vec 9 3"],
        available_actions: Bool[Array, "vec 9"],
        actions: Int[Array, "vec"],
        rewards: Float32[Array, "vec"],
        next_obs: Float32[Array, "vec 9 3"],
        done: Bool[Array, "vec"],
        importance: Float32[Array, "vec"],
        took_turn: Bool[Array, "vec"],
    ):
        params, metrics = self.update_model(
            params,
            obs,
            available_actions,
            actions,
            rewards,
            next_obs,
            done,
            importance,
            took_turn,
        )

        # set the importance back to 1 if it's the end of an episode
        importance = jnp.maximum(importance * self.discount, done)
        return params, metrics, importance
