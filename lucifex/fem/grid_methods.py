from typing import Literal, Callable, overload
from collections.abc import Iterable
from functools import singledispatch

from dolfinx.fem import Function
import numpy as np

from ..utils.array_utils import as_index
from ..utils.py_utils import as_slice, StrSlice, MultipleDispatchTypeError
from ..mesh.mesh2npy import GridMesh
from .fem2npy import as_grid_function, GridFunction


@overload
def grid_cross_section(
    fxyz: Function | GridFunction | tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> tuple[GridFunction, float]: 
    ...


@overload
def grid_cross_section(
    fxyz: Iterable[Function | GridFunction],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
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
    value: float | None = None,
    fraction: bool = True,
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
        value,
        fraction,
        axis_names,
    )


@grid_cross_section.register(tuple)
def _(
    fxyz: tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
): 
    values, *axes = fxyz
    f_grid = GridFunction(values, GridMesh(axes))
    return grid_cross_section(
        f_grid,
        axis,
        value,
        fraction,
        axis_names,
    )


@grid_cross_section.register(GridFunction)
def _(
    f_grid: GridFunction,
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
): 
    if fraction:
        f_fraction = value
        if f_fraction < 0 or f_fraction > 1:
            raise ValueError("Fraction must be in interval [0, 1]")
        f_value = None
    else:
        f_fraction = None
        f_value = value

    if not isinstance(axis, int):
        axis_index = axis_names.index(axis)
    else:
        axis_index = axis

    dim = len(f_grid.mesh.axes)
    if dim == 2:
        return _cross_section_line(f_grid, f_fraction, f_value, axis_index)
    elif dim == 3:
        return _cross_section_colormap(f_grid, f_fraction, f_value, axis_index)
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
    y_fraction: float | None,
    y_value: float | int | None,
    y_index: Literal[0, 1],
) -> tuple[GridFunction, float]:
    
    y_axis = u.mesh.axes[y_index]
    x_axis = u.mesh.axes[(y_index + 1) % 2]

    if y_value is not None:
        yaxis_index = as_index(y_axis, y_value)
    else:
        assert y_fraction is not None
        if np.isclose(y_fraction, 1):
            yaxis_index = -1
        else:
            yaxis_index = int(y_fraction * len(y_axis))
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
    z_fraction: float | None,
    z_value: float | int | None,
    z_index: Literal[0, 1, 2],
) -> tuple[GridFunction, float]:
    
    z_axis = u.mesh.axes[z_index]
    x_axis = u.mesh.axes[(z_index + 1) % 3]
    y_axis = u.mesh.axes[(z_index + 2) % 3]

    if z_value is not None:
        zaxis_index = as_index(z_axis, z_value)
    else:
        assert z_fraction is not None
        zaxis_index = int(z_fraction * len(z_axis))
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
def grid_average(
    u: Function | GridFunction,
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    use_cache: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> GridFunction | float:
    ...


@overload
def grid_average(
    u: Iterable[Function | GridFunction],
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    use_cache: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> list[GridFunction] | list[float]:
    ...



def grid_average(
    u: Function | GridFunction | Iterable[Function | GridFunction],
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    use_cache: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
):
    
    if not isinstance(u, (Function, GridFunction)):
        return [grid_average(i, axis, slc, use_cache, axis_names) for i in u]
    
    if isinstance(u, Function):
        u = as_grid_function(use_cache=use_cache)(u)
    
    if not isinstance(axis, tuple):
        axis = (axis, )
    _axis = tuple(axis_names.index(i) for i in axis)

    if not isinstance(slc, tuple):
        slc = [slc] * len(u.mesh.axes)
    _slc = tuple(as_slice(i) for i in slc)

    avg = _grid_average(u.value, _axis, *_slc)

    if isinstance(avg, float):
        return avg
    else:
        avg_axes = [ax[s] for i, (ax, s) in enumerate(zip(u.mesh.axes, _slc)) if not i in _axis]
        return GridFunction(avg, GridMesh(avg_axes))


def _grid_average(
    values: np.ndarray,
    axis: int | tuple[int, ...] | None,
    *slc: slice,
) -> np.ndarray | float:

    return np.mean(values[slc], axis)


def where_on_grid(
    f: GridFunction | Function,
    condition: Callable[[np.ndarray], np.ndarray],
    use_cache: bool | tuple[bool, bool] = True,
) -> tuple[np.ndarray, ...]:
    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
    use_mesh_cache, use_func_cache = use_cache
    f_grid = as_grid_function(use_cache=use_func_cache)(f, use_mesh_cache=use_mesh_cache)
    axes = f_grid.mesh.axes
    indices = np.where(condition(f_grid))
    return tuple(x[i] for x, i in zip(axes, indices))

