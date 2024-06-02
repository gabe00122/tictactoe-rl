import optax
from ..transformer import ActorHead, CriticHead, Transformer
from .agent_settings import AgentSettings
from .actor_critic import ActorCritic, ActorCriticModel


def create_actor_critic(settings: AgentSettings, total_steps: int) -> ActorCritic:
    actor_critic_model = ActorCriticModel(
        body=Transformer(token_features=32, num_layers=3),
        actor_head=ActorHead(actions=9),
        critic_head=CriticHead(),
    )

    optimizer = optax.adamw(
        optax.linear_schedule(settings["learning_rate"], 0, total_steps),
        b1=settings["adam_beta"],
        b2=settings["adam_beta"],
        weight_decay=settings["weight_decay"],
    )

    actor_critic = ActorCritic(
        model=actor_critic_model,
        optimizer=optimizer,
        discount=settings["discount"],
        actor_coef=settings["actor_coef"],
        entropy_coef=settings["entropy_coef"],
    )
    return actor_critic
