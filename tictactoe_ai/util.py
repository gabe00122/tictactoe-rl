from jax import random
from jaxtyping import PRNGKeyArray


def split_n(rng_key: PRNGKeyArray, num: int) -> tuple[PRNGKeyArray, PRNGKeyArray]:
    keys = random.split(rng_key, num + 1)
    return keys[0], keys[1:]
