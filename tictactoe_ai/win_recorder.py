from jaxtyping import Scalar, Int32

from typing import NamedTuple


class WinMetrics(NamedTuple):
    winX: Int32[Scalar, ""]
    winO: Int32[Scalar, ""]
    ties: Int32[Scalar, ""]
