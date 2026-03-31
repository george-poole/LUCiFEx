from typing import Literal, Callable, overload, Any
from collections.abc import Iterable
from functools import singledispatch
from scipy.interpolate import RegularGridInterpolator

from dolfinx.fem import Function
import numpy as np

from ..utils.array_utils import as_index
from ..utils.py_utils import (
    as_slice, StrSlice, OverloadTypeError, 
    optional_lru_cache,
)
from ..mesh.mesh2npy import GridMesh
from .fem2npy import as_grid_function, GridFunction


# TODO consistent overload with `Function | GridFunction -> GridFunction` and `tuple -> tuple`
@overload
def cross_section_grid(
    u: Function | GridFunction | tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> tuple[GridFunction, float]: 
    ...


@overload
def cross_section_grid(
    u: Iterable[Function | GridFunction],
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
    strict: bool = False,
) -> tuple[list[GridFunction], float | list[float]]: 
    ... 


@singledispatch
def cross_section_grid(f, *_, **__):
    raise OverloadTypeError(f)


@cross_section_grid.register(Function)
def _(
    u: Function,
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
): 
    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
    use_mesh_cache, use_func_cache = use_cache
    u_grid = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)
    return cross_section_grid(
        u_grid,
        axis,
        target,
        fraction,
        axis_names,
    )


@cross_section_grid.register(tuple)
def _(
    xyz: tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    target: int | float,
    fraction: bool = False,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
): 
    *axes, value = xyz
    u_grid = GridFunction(value, GridMesh(axes))
    return cross_section_grid(
        u_grid,
        axis,
        target,
        fraction,
        axis_names,
    )


@cross_section_grid.register(GridFunction)
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


@cross_section_grid.register(Iterable)
def _(
    u: list[Function | GridFunction],
    axis: str | Literal[0, 1, 2],
    target: float | None = None,
    fraction: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
    strict: bool = False,
): 
    _cross_sections = []
    _values = []
    pass_kws = lambda u: dict(use_cache=use_cache) if isinstance(u, Function) else dict()
    grid, target  = cross_section_grid(
        u[0], axis, target, fraction, axis_names, **pass_kws(u[0])
    )
    _cross_sections.append(grid)
    _values.append(target)

    for _u in u[1:]:
        _grid, _value  = cross_section_grid(
            _u, axis, target, fraction, axis_names, **pass_kws(_u),
        )
        if strict:
            assert np.isclose(target, _value)
            assert np.all(np.isclose(grid.mesh.axes, _grid.mesh.axes)) # FIXME typing
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
def _average_grid(
    u: Function | GridFunction,
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> GridFunction | float:
    ...


@overload
def _average_grid(
    u: Iterable[Function | GridFunction],
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | tuple[StrSlice, ...] = ':',
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> list[GridFunction] | list[float]:
    ...


# TODO slc: int | float
def _average_grid(
    u: Function | GridFunction | Iterable[Function | GridFunction],
    axis: str | int | tuple[str | int, ...],
    slc: StrSlice | int | float | tuple[StrSlice | int | float, ...] = ':',
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
):
    if not isinstance(u, (Function, GridFunction)):
        return [_average_grid(i, axis, slc, axis_names, use_cache) for i in u]
    
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

    _slc = []
    for s, ax in zip(slc, u.mesh.axes, strict=True):
        if isinstance(s, (float, int)):
            s = as_index(ax, s)
        else:
            s = as_slice(s)
        _slc.append(s)

    avg = _array_average(u.value, _axis, *_slc)

    if isinstance(avg, float):
        return avg
    else:
        avg_axes = [ax[s] for i, (ax, s) in enumerate(zip(u.mesh.axes, _slc)) if not i in _axis]
        return GridFunction(avg, GridMesh(avg_axes))


def _array_average(
    values: np.ndarray,
    axis: int | tuple[int, ...] | None,
    *slc: slice | int,
) -> np.ndarray | float:

    return np.mean(values[slc], axis)


average_grid = optional_lru_cache(_average_grid)


@optional_lru_cache
def resample_grid(
    u: GridFunction | Function,
    factor: int | float | tuple[int | float, ...],
    name: str | None = None,
    use_cache: bool | tuple[bool, bool] = True,
    **interpolator_kws: Any,
) -> GridFunction:
    """
    `factor > 1` to refine and `factor < 1` to coarsen
    """
    if isinstance(u, Function):
        if isinstance(use_cache, bool):
            use_cache = (use_cache, use_cache)
        use_mesh_cache, use_func_cache = use_cache
        u = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)

    if name is None:
        name = u.name
    
    axes = u.mesh.axes

    if isinstance(factor, (int, float)):
        factor = tuple([factor] * len(axes))

    axes_fine = tuple(
        np.linspace(np.min(ax), np.max(ax), int(n * len(ax))) 
        for ax, n in zip(axes, factor, strict=True)
    )
    interpolator = RegularGridInterpolator(axes, u.value, **interpolator_kws)
    u_fine_values = interpolator(tuple(np.meshgrid(*axes_fine))).T

    return GridFunction(
        u_fine_values,
        GridMesh(axes_fine),
        name,
    )


@overload
def mirror_grid(
    u: Function,
    axis: str | Literal[0, 1, 2],
    rescale: float = 1.0,
    name: str | None = None,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> GridFunction:
    ...


@overload
def mirror_grid(
    u: GridFunction,
    axis: str | Literal[0, 1, 2],
    rescale: float = 1.0,
    name: str | None = None,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> GridFunction:
    ...


@overload
def mirror_grid(
    xyz: tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    rescale: float = 1.0,
    name: str | None = None,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
) -> tuple[np.ndarray, ...]:
    ...


def mirror_grid(
    u: GridFunction | Function | tuple[np.ndarray, ...],
    axis: str | Literal[0, 1, 2],
    rescale: float = 1.0,
    name: str | None = None,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool | tuple[bool, bool] = True,
) -> GridFunction | tuple[np.ndarray, ...]: # TODO @overload
    
    if isinstance(u, Function):
        if isinstance(use_cache, bool):
            use_cache = (use_cache, use_cache)
        use_mesh_cache, use_func_cache = use_cache
        u_grid = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)
        return mirror_grid(u_grid, axis, rescale, name, axis_names)
    
    if isinstance(u, GridFunction):
        axes = u.mesh.axes
        value = u.value
        *axes_mirror, value_mirror = mirror_grid((*axes, value), axis, rescale, name, axis_names)
        return GridFunction(
            value_mirror,
            GridMesh(axes_mirror),
            name,
        )

    *axes, value = u

    if not isinstance(axis, int):
        axis_index = axis_names.index(axis)
    else:
        axis_index = axis

    axes_mirror = []
    for i, ax in enumerate(axes):
        if i == axis_index:
            if np.isclose(ax[0], 0.0):
                pos_to_neg = True
                axm = np.concatenate((-ax[::-1], ax[1:]))
            elif np.isclose(ax[-1], 0.0):
                pos_to_neg = False
                axm = np.concatenate((ax[:-1], -ax[::-1]))
            else:
                raise ValueError('Can only mirror axes about the origin.')
        else:
            axm = ax
        axes_mirror.append(axm)

    dim = len(axes_mirror)

    if pos_to_neg:
        slc_pos = tuple(
            slice(1, None) if i == axis_index else slice(None) for i in range(dim)
        )
        slc_neg = tuple(
            slice(None, None, -1) if i == axis_index else slice(None) for i in range(dim)
        )
        rsc_neg = rescale
        rsc_pos = 1.0
    else:
        slc_neg = tuple(
            slice(0, -1) if i == axis_index else slice(None) for i in range(dim)
        )
        slc_pos = tuple(
            slice(None, None, -1) if i == axis_index else slice(None) for i in range(dim)
        )
        rsc_neg = 1.0
        rsc_pos = rescale

    value_mirror = np.concatenate(
        (rsc_neg * value[slc_neg], rsc_pos * value[slc_pos]), 
        axis=axis_index,
    )

    return *axes_mirror, value_mirror


def crop_grid(
    u: GridFunction | Function,
    crop: Iterable[tuple[float | int, float | int] | None],
    name: str | None = None,
    use_cache: bool | tuple[bool, bool] = True,
) -> GridFunction:

    if isinstance(u, Function):
        if isinstance(use_cache, bool):
            use_cache = (use_cache, use_cache)
        use_mesh_cache, use_func_cache = use_cache
        u_grid = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)
        return crop_grid(u_grid, crop, name)
    
    value_crop = np.copy(u.value)
    axes_crop = []
    
    for i, (crp, ax) in enumerate(zip(crop, u.mesh.axes)):
        if crp is None:
            axes_crop.append(ax)
        else:
            slc_start, slc_stop = (as_index(ax, trg) for trg in crp)
            ax_slc = slice(slc_start, slc_stop + 1)
            axes_crop.append(ax[ax_slc])
            grid_slc = [slice(0, None)] * len(u.mesh.axes)
            grid_slc[i] = ax_slc
            value_crop = value_crop[tuple(grid_slc)]

    return GridFunction(
        value_crop,
        GridMesh(tuple(axes_crop)),
        name,
    )


def copy_grid(
    u: GridFunction | Function,
    slc: tuple[int | float | StrSlice, ...] | None = None,
    change: float | Iterable[float] | np.ndarray | None = None,
    name: str | None = None,
    use_cache: bool | tuple[bool, bool] = True,
) -> GridFunction:
    if isinstance(u, Function):
        if isinstance(use_cache, bool):
            use_cache = (use_cache, use_cache)
        use_mesh_cache, use_func_cache = use_cache
        u_grid = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)
        return copy_grid(u_grid, slc, change, name)
    
    value_copy = np.copy(u.value)
    
    if slc is None and change is None:
        return GridFunction(
            value_copy,
            u.mesh,
            name,
        ) 

    if slc is None:
        slc = [slice(0, None)] * len(u.mesh.axes)

    value_slc: list[slice | int] = []
    for s, ax in zip(slc, u.mesh.axes, strict=True):
        if isinstance(s, (float, int)):
            s = as_index(ax, s)
        else:
            s = as_slice(s)
        value_slc.append(s)

    value_copy[tuple(value_slc)] = change

    return GridFunction(
        value_copy,
        u.mesh,
        name,
    )



def where_on_grid(
    u: GridFunction | Function,
    condition: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
    use_cache: bool | tuple[bool, bool] = True,
) -> tuple[np.ndarray, ...]:
    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
    use_mesh_cache, use_func_cache = use_cache
    u_grid = as_grid_function(use_cache=use_func_cache)(u, use_mesh_cache=use_mesh_cache)
    axes = u_grid.mesh.axes
    indices = np.where(condition(u_grid))
    return tuple(x[i] for x, i in zip(axes, indices))

