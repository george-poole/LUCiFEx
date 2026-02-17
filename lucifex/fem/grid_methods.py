from typing import Literal, Callable, Iterable

from dolfinx.fem import Function
import numpy as np

from ..utils.array_utils import as_index
from ..utils.py_utils import as_slice, StrSlice
from ..mesh.mesh2npy import GridMesh
from .fem2npy import as_grid_function, GridFunction


def grid_cross_section(
    fxyz: Function | tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: tuple[bool, bool] = (True, False),
) -> tuple[GridFunction, float]:    
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

    if isinstance(fxyz, Function):
        use_mesh_cache, use_func_cache = use_cache
        f_grid = as_grid_function(use_cache=use_func_cache)(fxyz, use_mesh_cache=use_mesh_cache)
    else:
        values, *axes = fxyz
        f_grid = GridFunction(values, GridMesh(axes))

    dim = len(f_grid.mesh.axes)
    if dim == 2:
        return _cross_section_line(f_grid, f_fraction, f_value, axis_index)
    elif dim == 3:
        return _cross_section_colormap(f_grid, f_fraction, f_value, axis_index)
    else:
        raise ValueError(f'Cannot get a cross-section in d={dim}.')


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
        y_line = u.values[yaxis_index, :]
    elif y_index == 1:
        y_line = u.values[:, yaxis_index]
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
        z_grid = u.values[zaxis_index, :, :]
    elif z_index == 1:
        z_grid = u.values[:, zaxis_index, :]
    elif z_index == 2:
        z_grid = u.values[:, :, zaxis_index]
    else:
        raise ValueError
    
    u_cmap = GridFunction(
        z_grid,
        GridMesh((x_axis, y_axis)),
    )

    return u_cmap, z_value


def grid_cross_section_series(
    u: Iterable[Function | tuple[np.ndarray, ...]],
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    fraction: bool = True,
    use_cache: tuple[bool, bool] = (True, False),
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> list[GridFunction]:
    _cross_sections = []
    grid, value  = grid_cross_section(
        u[0], axis, value, fraction, axis_names, use_cache,
    )
    _cross_sections.append(grid)

    for _u in u[1:]:
        _grid, _value  = grid_cross_section(_u, axis, value, fraction, axis_names, use_cache)
        assert np.isclose(value, _value)
        assert np.all(np.isclose(grid.mesh.axes, _grid.mesh.axes))
        _cross_sections.append(_grid)

    return _cross_sections


def grid_average(
    u: Function  | np.ndarray,
    axis: str | int | None = None,
    slc: StrSlice | tuple[StrSlice, StrSlice] = ':',
) -> np.ndarray:
    if isinstance(u, Function):
        values = as_grid_function(u).values
    else:
        values = u
        
    if isinstance(axis, str):
        axis = ('x', 'y', 'z').index(axis)

    if isinstance(slc, tuple):
        slc_x, slc_y = slc
    else:
        slc_x, slc_y = slc, slc

    slc_x = as_slice(slc_x)
    slc_y = as_slice(slc_y)

    return np.mean(values[slc_x, slc_y], axis)


def where_on_grid(
    f: Function,
    condition: Callable[[np.ndarray], np.ndarray],
    use_cache: tuple[bool, bool] = (True, False),
) -> tuple[np.ndarray, ...]:
    use_mesh_cache, use_func_cache = use_cache
    f_grid = as_grid_function(use_cache=use_func_cache)(f, use_mesh_cache=use_mesh_cache)
    axes = f_grid.mesh.axes
    indices = np.where(condition(f_grid))
    return tuple(x[i] for x, i in zip(axes, indices))

