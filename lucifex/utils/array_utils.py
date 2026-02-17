import operator
from typing import Callable, Iterable, overload

import numpy as np

from .py_utils import StrSlice, as_slice


def derivative(
    y: Iterable[float],
    x: Iterable[float],
    order: int = 1,
    edge_order: int = 2,
):
    _dydx = np.gradient(y, x, edge_order=edge_order)
    for _ in range(order - 1):
        _dydx = np.gradient(_dydx, x, edge_order=edge_order)

    return _dydx


@overload
def moving_average(
    series: Iterable[float],
    window: int,
) -> list[float]:
    ...


@overload
def moving_average(
    series: Iterable[float],
    window: float,
    time_series: Iterable[float],
) -> list[float]:
    ...

def moving_average(
    series: Iterable[float],
    window: int | float,
    time_series: Iterable[float] | None = None,
):
    ma = [series[0]]
    if isinstance(window, int):
        ma.extend([np.mean(series[max(i - window, 0): i]) for i in range(1, len(series))])
    else:
        assert time_series is not None
        assert len(series) == len(time_series)
        for i in range(1, len(series)):
            t_target = time_series[i] - window
            lower_index = as_index(time_series, t_target)
            if lower_index == i:
                ma.append(series[i])
            else:
                ma.append(np.mean(series[max(lower_index, 0): i]))
            
    return ma 


def as_index(
    arr: np.ndarray | Iterable[float],
    target: int | float,
    fraction: bool = False,
    condition: Callable[[float, float], bool] | str | None = None,
    msg: str = 'Target validation condition not satisfied',
) -> int:
    if isinstance(target, int):
        return target
    
    if fraction:
        return int(target * len(arr))
    
    arr = np.sort(arr)
    
    if isinstance(condition, str):
        condition = getattr(operator, condition)

    arr_diff = np.abs([i - target for i in arr])           
    target_index = np.argmin(arr_diff)

    if condition is not None:
        approx = arr[target_index]
        if not condition(approx, target):
            for i in (-1, 1, -2, 2):
                if condition(approx, target):
                    return target_index + i
            raise ValueError(f'{msg}. target={target}, approx={approx}')

    return target_index


def as_indices(
    arr: np.ndarray | Iterable[float],
    targets: range | Iterable[int | float] | int | StrSlice,
    fraction: bool = False,
    condition: Callable[[float, float], bool] | str | None = None,
    window: bool = False,
) -> Iterable[int]:
    if isinstance(targets, range):
        indices = targets
    elif isinstance(targets, StrSlice):
        slc = as_slice(targets)
        indices = range(slc.start, slc.stop, slc.step)
    elif isinstance(targets, int):
        stop = len(arr)
        step = stop // targets
        indices = range(0, stop, step)
    else:
        indices = [as_index(arr, i, fraction, condition) for i in targets]
        if window:
            if indices[0] < targets[0]:
                indices[0] += 1
            if indices[1] > targets[-1]:
                indices[0] -= 1
    return indices
