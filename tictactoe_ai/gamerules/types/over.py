from typing import NamedTuple
from jaxtyping import Scalar, Bool, Int32


class OverResult(NamedTuple):
    game_state: Int32[Scalar, ""]  # ongoing, o won, tied, x won
