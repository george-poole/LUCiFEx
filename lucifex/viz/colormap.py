from functools import singledispatch
from typing import overload, Iterable

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from ufl.core.expr import Expr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.tri.triangulation import Triangulation

from ..mesh.cartesian import CellType
from ..utils import (is_scalar, grid, triangulation, fem_function, is_structured, 
                     filter_kwargs, MultipleDispatchTypeError, extract_mesh)

from .utils import LW, set_axes, optional_ax, set_axes, optional_fig_ax


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    f: Function,
    colorbar: bool | tuple[float, float] = True,
    use_cache: bool = False,
    **kwargs,
) -> None:
    """Plots a colormap of a scalar-valued function"""
    ...


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    expr: Expr,
    colorbar: bool | tuple[float, float] = True,
    use_cache: bool = False,
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    """Plots a colormap of a scalar-valued expression"""
    ...


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    fxy: tuple[np.ndarray, np.ndarray, np.ndarray],
    colorbar: bool | tuple[float, float] = True,
    structured: bool | None = None,
    **kwargs,
) -> None:
    ...


def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    f,
    colorbar = True,
    *args,
    **kwargs,
) -> None:
    return __plot_colormap(f, fig, ax, colorbar, *args, **kwargs)


@singledispatch
def __plot_colormap(f, *a, **k):
    raise MultipleDispatchTypeError(f, __plot_colormap)


@__plot_colormap.register(Function)
def _(
    f: Function,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float],
    use_cache: bool = False,
    **kwargs,
) -> tuple[Figure, Axes]:
    if not is_scalar(f):
        raise ValueError("Colormap plotting is for scalar-valued functions only.")

    mesh = f.function_space.mesh
    cell_type = mesh.topology.cell_name()
    structured = is_structured(use_cache=True)(mesh)

    match cell_type, structured:
        case CellType.TRIANGLE, False:
            trigl = triangulation(use_cache=True)(f.function_space.mesh)
            x, y = trigl.x, trigl.y
            f_np = triangulation(use_cache=use_cache)(f)
        case CellType.TRIANGLE | CellType.QUADRILATERAL, True:
            x, y = grid(use_cache=True)(f.function_space.mesh)
            f_np = grid(use_cache=use_cache)(f)
        case CellType.QUADRILATERAL, False:
            raise NotImplementedError(
                """ 
            Plotting functions on unstructured quadrilateral meshes 
            is not supported by the `matplotlib` backend. Consider using
            `pyvista` instead. """
            )
        case _:
            raise ValueError

    return __plot_colormap(
        (f_np, x, y), fig, ax, colorbar, structured, **kwargs
    )


@__plot_colormap.register(Expr)
def _(
    expr: Expr,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float],
    use_cache: bool = False,
    mesh: Mesh | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    if mesh is None:
        mesh = extract_mesh(expr)
    func = fem_function((mesh, 'P', 1), expr, use_cache=use_cache)
    return __plot_colormap(
        func,
        fig, 
        ax,
        colorbar,
        **kwargs
    )


@__plot_colormap.register(tuple)
def _(
    fxy: tuple[np.ndarray, np.ndarray, np.ndarray],
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float],
    structured: bool | None = None,
    **kwargs,
):
    f, x, y = fxy

    _plt_kwargs = dict(cmap="hot", shading="gouraud")
    _axs_kwargs = dict(x_lims=x, y_lims=y, x_label='x', y_label='y', aspect='equal', tex=True)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    filter_kwargs(set_axes)(ax, **_kwargs)

    if structured is None:
        structured = bool(len(x) == len(np.unique(x)) and len(y) == len(np.unique(y)))

    if not structured:
        cmap = filter_kwargs(ax.tripcolor)(x, y, f, **_kwargs)
    else:
        cmap = filter_kwargs(ax.pcolormesh)(x, y, f.T, **_kwargs)

    if colorbar is not False:
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(cmap, ax_cbar)
        if isinstance(colorbar, tuple):
            assert len(colorbar) == 2
            cmap.set_clim(*colorbar)


# FIXME syntax highlight as function not variable
plot_colormap = optional_fig_ax(_plot_colormap)


# TODO singledispatch version
@optional_ax
def plot_contours(
    ax: Axes,
    f: Function | tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, Triangulation],
    levels: Iterable[float] | None = None,
    use_cache: bool = False,
    **kwargs,
) -> None:
    """Plots contours of a scalar-valued function"""

    if isinstance(f, tuple):
        _plt_kwargs = dict(linestyles="solid", color="black", linewidths=LW)
        _axs_kwargs = dict(x_label='x', y_label='y', aspect='equal', tex=True)
        match f:
            case fxy, trigl:
                _axs_kwargs.update(x_lims=trigl.x, y_lims=trigl.y)
                tri = True
            case fxy, x, y:
                _axs_kwargs.update(x_lims=x, y_lims=y)
                tri = False
            case _:
                raise ValueError

        _kwargs = _plt_kwargs | _axs_kwargs
        _kwargs.update(**kwargs)
        filter_kwargs(set_axes)(ax, **_kwargs)

        if tri:
            filter_kwargs(ax.tricontour, ContourSet)(trigl, fxy, levels=levels, **_kwargs)
        else:
            filter_kwargs(ax.contour, ContourSet)(x, y, fxy.T, levels=levels, **_kwargs)

    else:
        structured = is_structured(use_cache=use_cache)(f.function_space.mesh)
        if not structured:
            trigl = triangulation(use_cache=True)(f.function_space.mesh)
            f_trigl = triangulation(use_cache=use_cache)(f)
            return plot_contours((f_trigl, trigl), levels, **_kwargs)
        else:
            x, y = grid(use_cache=True)(f.function_space.mesh)
            f_grid = grid(use_cache=use_cache)(f)
            return plot_contours((f_grid, x, y), levels, **_kwargs)
        
