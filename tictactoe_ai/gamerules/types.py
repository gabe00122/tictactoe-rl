from typing import NamedTuple
from jaxtyping import Int8, Array, Scalar


class GameState(NamedTuple):
    board: Int8[Array, "3 3"]
    active_player: Int8[Scalar, ""]
    over_result: Int8[Scalar, ""]


class VectorizedGameState(NamedTuple):
    board: Int8[Array, "vec 3 3"]
    active_player: Int8[Scalar, "vec"]
    over_result: Int8[Scalar, "vec"]
