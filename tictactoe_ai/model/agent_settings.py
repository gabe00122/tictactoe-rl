from typing import TypedDict, Literal


class AgentSettings(TypedDict):
    type: Literal["actor_critic"]
    discount: float
    root_hidden_layers: list[int]
    actor_hidden_layers: list[int]
    critic_hidden_layers: list[int]
    learning_rate: float
    adam_beta: float
    weight_decay: float
    actor_coef: float
    entropy_coef: float
