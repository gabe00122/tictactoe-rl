import jax
from jaxtyping import Scalar, Int32
from beartype import beartype
from typing import NamedTuple
from .gamerules.over import OverResult


class WinMetrics(NamedTuple):
    winX: Int32[Scalar, ""]
    winO: Int32[Scalar, ""]
    ties: Int32[Scalar, ""]


@jax.vmap
@beartype
def record_wins(result: OverResult) -> WinMetrics:
    pass
