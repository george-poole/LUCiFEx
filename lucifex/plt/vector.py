from typing import Callable

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.quiver import Quiver

from ..fem import TriFunction, GridFunction, as_npy_function
from ..utils.fenicsx_utils import (
    is_vector, create_function, extract_mesh, ShapeError, 
    extract_component_functions, as_function,
)
from ..utils.py_utils import create_kws_filterer
from .utils import set_axes, optional_ax


@optional_ax
def plot_quiver(
    ax: Axes,
    u: Function | Expr | tuple[Function, Function] 
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    arrow_slc: int | tuple[int, int] = 1,
    use_cache: tuple[bool, bool] = True,
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    """
    Plots quiver arrows of a two-dimensional vector
    """
    if isinstance(u, tuple) and len(u) == 4:
        return _plot_quiver(ax, u, arrow_slc, **kwargs)

    if isinstance(u, tuple) and len(u) == 2:
        x_y_ux_uy = _extract_x_y_ux_uy_arrays(
            u, use_cache, mesh,
        )
        return _plot_quiver(ax, x_y_ux_uy, arrow_slc, **kwargs)
    
    ux, uy = _extract_xy_functions(u, mesh)
    return plot_quiver(ax, (ux, uy), arrow_slc, **kwargs)


def _plot_quiver(
    ax: Axes,
    x_y_ux_uy: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    arrow_slc: int | tuple[int, int],
    **kwargs,
) -> None:
    
    x, y, ux, uy = x_y_ux_uy
    _kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _kwargs.update(kwargs)
    create_kws_filterer(set_axes)(ax, **_kwargs)

    if len(np.shape(x)) ==  2:
        tri = True
    else:
        tri = False

    if tri:
        x_quiv = x[::arrow_slc]
        y_quiv = y[::arrow_slc]
        ux_quiv = ux[::arrow_slc]
        uy_quiv = uy[::arrow_slc]
    else:
        if isinstance(arrow_slc, int):
            arrow_slc = (arrow_slc, arrow_slc)
        x_arrow_slc, y_arrow_slc = arrow_slc
        x_quiv = x[::x_arrow_slc]
        y_quiv = y[::y_arrow_slc]
        ux_quiv = ux[::x_arrow_slc, ::y_arrow_slc].T
        uy_quiv = uy[::x_arrow_slc, ::y_arrow_slc].T

    create_kws_filterer(ax.quiver, Quiver)(
        x_quiv,
        y_quiv,
        ux_quiv,
        uy_quiv,
        **_kwargs,
    )


@optional_ax
def plot_streamlines(
    ax: Axes,
    u: Function | Expr 
    | tuple[Function | GridFunction | TriFunction | Expr, Function | GridFunction | TriFunction | Expr] 
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    density: float = 1.0,
    color: str | tuple[str, Callable]= 'black',
    use_cache: bool | tuple[bool, bool] = True,
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    """
    Plots streamlines of a two-dimensional vector
    """
    if isinstance(u, tuple) and len(u) == 4:
        return _plot_streamlines(ax, u, density, color, **kwargs)
    
    if isinstance(u, tuple) and len(u) == 2:
        x_y_ux_uy = _extract_x_y_ux_uy_arrays(
            u, use_cache, mesh,
        )
        return _plot_streamlines(ax, x_y_ux_uy, density, color, **kwargs)
            
    ux, uy = _extract_xy_functions(u, mesh)
    return plot_streamlines(ax, (ux, uy), density, color, **kwargs)


def _plot_streamlines(
    ax: Axes,
    x_y_ux_uy: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    density,
    color: str | tuple[str, Callable],
    **kwargs,
) -> None:
    x, y, ux, uy = x_y_ux_uy

    if isinstance(color, str):
        color_func = lambda fx, fy: np.sqrt((fx) ** 2 + (fy) ** 2)
    else:
        color, color_func = color

    _axs_kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _axs_kwargs.update(kwargs)
    create_kws_filterer(set_axes)(ax, **_axs_kwargs)

    if color in list(mpl_colormaps):
        norm = color_func(ux.T, uy.T)
        ax.streamplot(
            x, y, ux.T, uy.T, density=density, color=norm, cmap=color
        )
    else:
        ax.streamplot(x, y, ux.T, uy.T, density=density, color=color)


def _extract_xy_functions(
    u: Function | GridFunction | Expr,
    mesh: Mesh | None,
) -> tuple[Function, Function]:
    if isinstance(u, Expr) and not isinstance(u, Function):
        if mesh is None:
            mesh = extract_mesh(u)
        u = create_function((mesh, 'P', 1, 2), u)
    
    if not is_vector(u, dim=2):
        raise ShapeError(u, (2, ))
    
    return extract_component_functions(('P', 1), u)


def _extract_x_y_ux_uy_arrays(
    u: tuple[
        Function | GridFunction | TriFunction | Expr, 
        Function | GridFunction | TriFunction | Expr,
    ], 
    use_cache: bool | tuple[bool, bool],
    mesh: Mesh | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    u_npy: list[GridFunction | TriFunction] = []
    for ui in u:
        if isinstance(ui, (Function, Expr)):
            ui_p1 = as_function(('P', 1), ui)
            ui_npy = as_npy_function(ui_p1, use_cache=use_cache, mesh=mesh)
            u_npy.append(ui_npy)
        else:
            u_npy.append(ui)

    ux_npy, uy_npy = u_npy

    if type(ux_npy) is not type(uy_npy):
        raise TypeError('Vector components should both be of same type.') 
    
    if isinstance(ux_npy, TriFunction):
        triangles = ux_npy.mesh.cells
        x = ux_npy.mesh.x_coordinates[triangles]
        y = ux_npy.mesh.y_coordinates[triangles]
        ux_arr = ux_npy.value[triangles]
        uy_arr = uy_npy.value[triangles]
    
    if isinstance(ux_npy, GridFunction):
        x, y = ux_npy.mesh.axes
        ux_arr = ux_npy.value
        uy_arr = uy_npy.value

    return x, y, ux_arr, uy_arr


# def _x_y_fx_fy_arrays(
#     fx: Function, 
#     fy: Function,
#     use_cache: bool | tuple[bool, bool],
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

#     fx_np = as_npy_function(fx, use_cache=use_cache)
#     fy_np = as_npy_function(fy, use_cache=use_cache)
    
#     if isinstance(fx_np, TriFunction):
#         triangles = fx_np.mesh.cells
#         x = fx_np.mesh.x_coordinates[triangles]
#         y = fx_np.mesh.y_coordinates[triangles]
#         fx_new = fx_np.value[triangles]
#         fy_new = fy_np.value[triangles]
    
#     if isinstance(fx_np, GridFunction):
#         x, y = fx_np.mesh.axes
#         fx_new = fx_np.value
#         fy_new = fy_np.value

#     return x, y, fx_new, fy_new

