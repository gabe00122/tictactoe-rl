from jax import random
from jaxtyping import PRNGKeyArray
import random as py_random


def split_n(rng_key: PRNGKeyArray, num: int) -> tuple[PRNGKeyArray, PRNGKeyArray]:
    keys = random.split(rng_key, num + 1)
    return keys[0], keys[1:]


def generate_seed() -> int:
    return py_random.getrandbits(63)
