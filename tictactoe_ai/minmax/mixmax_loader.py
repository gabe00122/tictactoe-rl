from pathlib import Path
from jax import numpy as jnp
from .minmax_player import MinmaxAgent, MinmaxState
from .minmax_gen import generate_minmax


def get_minmax_agent(cache_path: str | Path = "./minmax_cache.npy") -> tuple[MinmaxAgent, MinmaxState]:
    cache_path = Path(cache_path)
    agent = MinmaxAgent()

    if cache_path.exists():
        state = agent.load(cache_path)
        print("Loaded minmax cache")
        return agent, state
    else:
        print("Creating a minmax cache")
        cache = generate_minmax()
        print(f"Saving minmax cache to {cache_path}")
        jnp.save(cache_path, cache)

        return agent, MinmaxState(cache)
