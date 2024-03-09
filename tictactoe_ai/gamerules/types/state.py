from typing import TypedDict
from jaxtyping import Array, Scalar, Int8


class GameState(TypedDict):
    board: Int8[Array, "3 3"]
    active_player: Int8[Scalar, ""]
    over_result: Int8[Scalar, ""]
