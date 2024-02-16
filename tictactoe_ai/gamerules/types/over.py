from typing import TypedDict
from jaxtyping import Scalar, Bool, Int8


class OverResult(TypedDict):
    is_over: Bool[Scalar, ""]
    winner: Int8[Scalar, ""]  # 0 is a draw
