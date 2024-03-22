import json
from typing import TypedDict


class RunSettings(TypedDict):
    seed: int
    total_steps: int
    env_num: int
    discount: float
    root_hidden_layers: list[int]
    actor_hidden_layers: list[int]
    critic_hidden_layers: list[int]
    learning_rate: float
    adam_beta: float
    weight_decay: float
    actor_coef: float
    entropy_coef: float


def save_settings(path, settings):
    with open(path, "w") as file:
        json.dump(settings, file, indent=2)


def load_settings(path):
    with open(path, "r") as file:
        settings = json.load(file)
    return settings
