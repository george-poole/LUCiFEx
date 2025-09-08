from typing import Callable, Sequence

import dolfinx as dfx
import numpy as np

from ..utils import is_structured
from .cartesian import CellType
from .utils import overload_mesh


def zeta_transform(
    s: float | Sequence[float],
    targets: Sequence[float],
    Lx: float | tuple[float, float],
) -> Callable:
    """ TODO update 
    Defines the stretching function

    `ζ(x) = x̄ + x̂·tanh(s·(x - x̄)) / tanh(s·x̂)`

    where 

    `x̄ = (x₋ + x₊) / 2` \\
    `x̂ = (x₊ - x₋) / 2`

    `s > 0` is the stretching strength 
    and `[x₋, x₊]` defines the interval.
    """

    targets = sorted(targets)

    if isinstance(Lx, float):
        x_min, x_max = 0.0, Lx
    else:
        x_min, x_max = Lx

    if not np.isclose(x_min, targets[0]):
        # inserting an artificial target below the range [xmin, xmax]
        targets.insert(0, x_min - targets[0])

    if not np.isclose(x_max, targets[-1]):
        # inserting an artificial target above the [xmin, xmax]
        targets.insert(-1, x_max + targets[-1])

    # pairs of targets splitting up the interval into piecewise intervals
    x_intervals = [(targets[i], targets[i + 1]) for i in range(len(targets) - 1)]

    if isinstance(s, (int, float)):
        stretches = [s] * len(x_intervals)

    def interval_tanh(
        x_min: float,
        x_max: float,
        s: float,
    ):
        xbar = 0.5 * (x_min + x_max)
        xhat = 0.5 * (x_max - x_min)
        return lambda x: xbar + xhat * np.tanh(s * (x - xbar)) / np.tanh(s * xhat)

    interval_funcs = [
        lambda x: interval_tanh(x_min, x_max, s)(x)
        for (x_min, x_max), s in zip(x_intervals, stretches)
    ]

    return lambda x: piecewise_func(x, x_intervals, interval_funcs)


def piecewise_func(
    x: np.ndarray,
    x_intervals: Sequence[tuple[float, float]],
    interval_funcs: Sequence[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    conditions = []
    for i, (x_min, x_max) in enumerate(x_intervals):
        if i == 0:
            conditions.append(((x >= x_min) & (x <= x_max)))
        else:
            conditions.append(((x > x_min) & (x <= x_max)))
    return np.piecewise(x, conditions, interval_funcs)