from flax import linen as nn
import optax
from .mlp import Mlp
from .run_settings import RunSettings
from .actor_critic import ActorCritic, ActorCriticModel


def create_actor_model(settings: RunSettings, action_space: int) -> nn.Module:
    return Mlp(
        features=settings["actor_hidden_layers"] + [action_space],
        # last_layer_scale=settings["actor_last_layer_scale"],
    )


def create_critic_model(settings: RunSettings) -> nn.Module:
    return Mlp(
        features=settings["critic_hidden_layers"] + [1],
        # last_layer_scale=settings["critic_last_layer_scale"],
    )


def create_actor_critic(settings: RunSettings) -> ActorCritic:
    body_model = Mlp(features=settings["root_hidden_layers"])
    actor_model = create_actor_model(settings, action_space=9)
    critic_model = create_critic_model(settings)

    actor_critic_model = ActorCriticModel(
        body=body_model, actor_head=actor_model, critic_head=critic_model
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
