from typing import Literal, Callable, overload
from collections.abc import Iterable
from functools import singledispatch
from scipy.interpolate import RegularGridInterpolator

from dolfinx.fem import Function
import numpy as np

from ..utils.array_utils import as_index
from ..utils.py_utils import (
    as_slice, StrSlice, MultipleDispatchTypeError, 
    optional_lru_cache,
)
from ..mesh.mesh2npy import GridMesh
from .fem2npy import as_grid_function, GridFunction


@overload
def grid_cross_section(
    fxyz: Function | GridFunction | tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> tuple[GridFunction, float]: 
    ...


@overload
def grid_cross_section(
    fxyz: Iterable[Function | GridFunction],
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
    strict: bool = False,
) -> tuple[list[GridFunction], float | list[float]]: 
    ... 


@singledispatch
def grid_cross_section(f, *_, **__):
    raise MultipleDispatchTypeError(f)


@grid_cross_section.register(Function)
def _(
    fxyz: Function,
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
): 
    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
    use_mesh_cache, use_func_cache = use_cache
    f_grid = as_grid_function(use_cache=use_func_cache)(fxyz, use_mesh_cache=use_mesh_cache)
    return grid_cross_section(
        f_grid,
        axis,
        target,
        fraction,
        axis_names,
    )


@grid_cross_section.register(tuple)
def _(
    fxyz: tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
): 
    values, *axes = fxyz
    f_grid = GridFunction(values, GridMesh(axes))
    return grid_cross_section(
        f_grid,
        axis,
        target,
        fraction,
        axis_names,
    )


@grid_cross_section.register(GridFunction)
def _(
    f_grid: GridFunction,
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
): 
    if not isinstance(axis, int):
        axis_index = axis_names.index(axis)
    else:
        axis_index = axis

    dim = len(f_grid.mesh.axes)
    if dim == 2:
        return _cross_section_line(f_grid, target, fraction, axis_index)
    elif dim == 3:
        return _cross_section_colormap(f_grid, target, fraction, axis_index)
    else:
        raise ValueError(f'Cannot get a cross-section in d={dim}.')


@grid_cross_section.register(Iterable)
def _(
    u: list[Function | GridFunction],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
    strict: bool = False,
): 
    _cross_sections = []
    _values = []
    grid, value  = grid_cross_section(
        u[0], axis, value, fraction, axis_names, use_cache,
    )
    _cross_sections.append(grid)
    _values.append(value)

    for _u in u[1:]:
        _grid, _value  = grid_cross_section(_u, axis, value, fraction, axis_names, use_cache)
        if strict:
            assert np.isclose(value, _value)
            assert np.all(np.isclose(grid.mesh.axes, _grid.mesh.axes))
        _cross_sections.append(_grid)
        _values.append(_value)

    if all(np.isclose(i, _values[0]) for i in _values):
        _values = _values[0]

    return _cross_sections, _values


def _cross_section_line(
    u: GridFunction,
    y_target: int | float,
    fraction: bool,
    y_index: Literal[0, 1],
) -> tuple[GridFunction, float]:
    
    y_axis = u.mesh.axes[y_index]
    x_axis = u.mesh.axes[(y_index + 1) % 2]

    yaxis_index = as_index(y_axis, y_target, fraction)
    y_value = y_axis[yaxis_index]

    if y_index == 0:
        y_line = u.value[yaxis_index, :]
    elif y_index == 1:
        y_line = u.value[:, yaxis_index]
    else:
        raise ValueError
    
    u_line = GridFunction(
        y_line, 
        GridMesh((x_axis, )),
    )

    return u_line, y_value


def _cross_section_colormap(
    u: GridFunction,
    z_target: int | float,
    fraction: bool,
    z_index: Literal[0, 1, 2],
) -> tuple[GridFunction, float]:
    
    z_axis = u.mesh.axes[z_index]
    x_axis = u.mesh.axes[(z_index + 1) % 3]
    y_axis = u.mesh.axes[(z_index + 2) % 3]

    zaxis_index = as_index(z_axis, z_target, fraction)
    z_value = z_axis[zaxis_index]

    if z_index == 0:
        z_grid = u.value[zaxis_index, :, :]
    elif z_index == 1:
        z_grid = u.value[:, zaxis_index, :]
    elif z_index == 2:
        z_grid = u.value[:, :, zaxis_index]
    else:
        raise ValueError
    
    u_cmap = GridFunction(
        z_grid,
        GridMesh((x_axis, y_axis)),
    )

    return u_cmap, z_value


@overload
def _grid_average(
    u: Function | GridFunction,
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> GridFunction | float:
    ...


@overload
def _grid_average(
    u: Iterable[Function | GridFunction],
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> list[GridFunction] | list[float]:
    ...


def _grid_average(
    u: Function | GridFunction | Iterable[Function | GridFunction],
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
):
    if not isinstance(u, (Function, GridFunction)):
        return [_grid_average(i, axis, slc, use_func_cache, axis_names) for i in u]
    
    if isinstance(u, Function):
        if isinstance(use_cache, bool):
            use_cache = (use_cache, use_cache)
        use_mesh_cache, use_func_cache = use_cache
        u = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)
    
    if not isinstance(axis, tuple):
        axis = (axis, )
    _axis = tuple(axis_names.index(i) for i in axis)

    if not isinstance(slc, tuple):
        slc = [slc] * len(u.mesh.axes)
    _slc = tuple(as_slice(i) for i in slc)

    avg = _array_average(u.value, _axis, *_slc)

    if isinstance(avg, float):
        return avg
    else:
        avg_axes = [ax[s] for i, (ax, s) in enumerate(zip(u.mesh.axes, _slc)) if not i in _axis]
        return GridFunction(avg, GridMesh(avg_axes))


def _array_average(
    values: np.ndarray,
    axis: int | tuple[int, ...] | None,
    *slc: slice,
) -> np.ndarray | float:

    return np.mean(values[slc], axis)


grid_average = optional_lru_cache(_grid_average)


@optional_lru_cache
def grid_resample(
    u: GridFunction,
    factor: int | float | tuple[int | float, ...],
    name: str | None = None,
    **kwargs,
) -> GridFunction:
    """
    `factor > 1` to refine and `factor < 1` to coarsen
    """
    if name is None:
        name = u.name
    
    axes = u.mesh.axes

    if isinstance(factor, (int, float)):
        factor = tuple([factor] * len(axes))

    axes_fine = tuple(
        np.linspace(np.min(ax), np.max(ax), int(n * len(ax))) 
        for ax, n in zip(axes, factor, strict=True)
    )
    interpolator = RegularGridInterpolator(axes, u.value, **kwargs)
    u_fine_values = interpolator(tuple(np.meshgrid(*axes_fine))).T

    return GridFunction(
        u_fine_values,
        GridMesh(axes_fine),
        name,
    )


def grid_mirror(
    u: GridFunction | Function,
    axis: str | Literal[0, 1, 2],
    name: str | None = None,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> GridFunction:
    if isinstance(u, Function):
        if isinstance(use_cache, bool):
            use_cache = (use_cache, use_cache)
        use_mesh_cache, use_func_cache = use_cache
        u = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)

    if not isinstance(axis, int):
        axis_index = axis_names.index(axis)
    else:
        axis_index = axis

    axes_mirror = []
    for i, axis in enumerate(u.mesh.axes):
        if i == axis_index:
            if np.isclose(axis[0], 0.0):
                pos_to_neg = True
                axm = np.array([*axis[::-1], *axis])
            elif np.isclose(axis[-1], 0.0):
                pos_to_neg = False
                axm = np.array([*axis, *axis[::-1]])
            else:
                raise ValueError('Can only mirror axes about the origin.')
        else:
            axm = axis
        axes_mirror.append(axm)

    u_mirror_shape = tuple(2 * n - 1 if i == axis_index else n for i, n in enumerate(u.npy_shape))
    u_mirror_values = np.zeros(u_mirror_shape)

    # TODO
    i_mid = ...
    if pos_to_neg:
        slc_pos = tuple(slice(i_mid, None) if i == axis_index else slice(None) for i in ...)
        slc_neg = tuple()
    else:
        ...
    
    u_mirror_values[slc_pos] = u.value
    u_mirror_values[slc_neg] = u.value

    return GridFunction(
        u_mirror_values,
        GridMesh(axes_mirror),
        name,
    )



def where_on_grid(
    u: GridFunction | Function,
    condition: Callable[[np.ndarray], np.ndarray],
    use_cache: bool | tuple[bool, bool] = True,
) -> tuple[np.ndarray, ...]:
    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
    use_mesh_cache, use_func_cache = use_cache
    u_grid = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)
    axes = u_grid.mesh.axes
    indices = np.where(condition(u_grid))
    return tuple(x[i] for x, i in zip(axes, indices))

