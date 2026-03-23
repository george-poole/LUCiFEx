import operator
from typing import Callable, Iterable, Any, overload

import numpy as np
from scipy.interpolate import PchipInterpolator

from .py_utils import StrSlice, as_slice


def derivative(
    y: Iterable[float],
    x: Iterable[float],
    order: int = 1,
    edge_order: int = 2,
) -> np.ndarray:
    """
    `dy/dx`
    """
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


@overload
def as_index(
    arr: Iterable[float],
    target: int | float,
    fraction: bool = False,
    condition: Callable[[float, float], bool] | str | None = None,
    msg: str = 'Target validation condition not satisfied',
) -> int:
    ...


@overload
def as_index(
    arr: Iterable[float],
    target: Iterable[int | float] | range | StrSlice | int,
    fraction: bool = False,
    condition: Callable[[float, float], bool] | str | None = None,
    msg: str = 'Target validation condition not satisfied',
    window: bool = False,
    int_as_range: bool = False,
) -> Iterable[int]:
    ...


def as_index(
    arr: Iterable[float],
    target,
    fraction: bool = False,
    condition: Callable[[float, float], bool] | str | None = None,
    msg: str = 'Target validation condition not satisfied',
    window: bool = False,
    int_as_range: bool = False,
) -> Iterable[int]:
    if isinstance(target, float):
        return _as_index(arr, target, fraction, condition, msg)
    
    if not int_as_range and isinstance(target, int):
        return _as_index(arr, target, fraction, condition, msg)

    if isinstance(target, range):
        return target
    
    if isinstance(target, StrSlice):
        slc = as_slice(target)
        return range(slc.start, slc.stop, slc.step)
    
    if int_as_range and isinstance(target, int):
        stop = len(arr)
        step = stop // target
        return range(0, stop, step)
    
    indices = [as_index(arr, i, fraction, condition) for i in target]
    if window:
        if indices[0] < target[0]:
            indices[0] += 1
        if indices[1] > target[-1]:
            indices[0] -= 1

    return indices


def _as_index(
    arr: Iterable[float],
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


def resample(
    y: Iterable[float],
    x: Iterable[float],
    x_new: Iterable[float],
    interpolator: Callable | str = PchipInterpolator
) -> np.ndarray:
    if isinstance(interpolator, str):
        import scipy.interpolate as spi
        interpolator = getattr(spi, interpolator)
    return interpolator(x, y)(x_new)