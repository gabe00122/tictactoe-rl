from flax import linen as nn
import optax
from .mlp import MlpBody, ActorHead, CriticHead
from .run_settings import RunSettings
from .actor_critic import ActorCritic, ActorCriticModel


def create_actor_critic(settings: RunSettings) -> ActorCritic:
    actor_critic_model = ActorCriticModel(
        body=MlpBody(features=settings["root_hidden_layers"]),
        actor_head=ActorHead(actions=9),
        critic_head=CriticHead()
    )

    optimizer = optax.adamw(
        optax.linear_schedule(settings["learning_rate"], 0, settings["total_steps"]),
        b1=settings["adam_beta"],
        b2=settings["adam_beta"],
        weight_decay=settings["weight_decay"],
    )

    actor_critic = ActorCritic(
        model=actor_critic_model,
        optimizer=optimizer,
        discount=settings["discount"],
        actor_coef=settings["actor_coef"],
        critic_coef=settings["critic_coef"],
    )
    return actor_critic
