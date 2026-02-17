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
from matplotlib.cm import ScalarMappable

from ..fem import GridFunction, TriFunction, as_npy_function
from ..utils.fenicsx_utils import (
    is_scalar, create_function, 
    is_grid, extract_mesh,
)
from ..utils.py_utils import filter_kwargs, MultipleDispatchTypeError

from .utils import LW, set_axes, optional_ax, set_axes, optional_fig_ax, optional_fig_axs


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    f: Function | Expr,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    use_cache: bool | tuple[bool, bool] = (True, False),
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    ...


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    u: GridFunction | TriFunction,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    use_cache: bool | tuple[bool, bool] = (True, False),
    **kwargs,
) -> None:
    ...


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Triangulation, np.ndarray],
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    **kwargs,
) -> None:
    ...


@singledispatch
def _plot_colormap(arg, *_, **__):
    raise MultipleDispatchTypeError(arg, _plot_colormap)


@_plot_colormap.register(Figure)
def _(
    fig: Figure,
    ax: Axes,
    arg,
    *args,
    **kwargs,
):
    return _plot_colormap(arg, fig, ax, *args, **kwargs)


@_plot_colormap.register(Function)
@_plot_colormap.register(Expr)
def _(
    u: Function | Expr,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    use_cache: bool | tuple[bool, bool] = (True, False),
    mesh: Mesh | None = None,
    **kwargs,
):
    if not is_scalar(u):
        raise ValueError("Colormap plots must be of scalar-valued quantities.")
    u_npy = as_npy_function(u, grid, use_cache, mesh)
    return _plot_colormap(
        u_npy, fig, ax, colorbar, grid, **kwargs
    )


@_plot_colormap.register(GridFunction)
def _(
    u: GridFunction,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    **kwargs,
):
    xyz = (*u.mesh.axes, u.values)
    return _plot_colormap(
        xyz, fig, ax, colorbar, grid, **kwargs,
    )


@_plot_colormap.register(TriFunction)
def _(
    u: TriFunction,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    **kwargs,
):
    xyz = (u.mesh.x_coordinates, u.mesh.y_coordinates, u.values)
    return _plot_colormap(
        xyz, fig, ax, colorbar, grid, triang=u.mesh.triangulation, **kwargs,
    )


@_plot_colormap.register(tuple)
def _(
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    triang: Triangulation | None = None,
    **kwargs,
):
    x, y, z = xyz

    _plt_kwargs = dict(cmap="hot", shading="gouraud")
    _axs_kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)
    filter_kwargs(set_axes)(ax, **_kwargs)

    if isinstance(colorbar, tuple):
        vmin, vmax = colorbar
        _kwargs.update(vmin=vmin, vmax=vmax)

    if triang is not None:
        cmap = filter_kwargs(ax.tripcolor)(triang, z, **_kwargs)
    else:
        if grid is None:
            grid = is_grid((x, y))
        if grid:
            cmap = filter_kwargs(ax.pcolormesh)(x, y, z.T, **_kwargs)
        else:
            cmap = filter_kwargs(ax.tripcolor, ('triangles', 'mask'))(x, y, z, **_kwargs)

    if colorbar is not False:
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(cmap, ax_cbar)


plot_colormap = optional_fig_ax(_plot_colormap)


# @overload
# def _plot_contours(
#     ax: Axes,
#     f: Function,
#     levels: Iterable[float] | int | None = None,
#     use_cache: bool | tuple[bool, bool] = (True, False),
#     **kwargs,
# ) -> None:
#     ...


# @overload
# def _plot_contours(
#     ax: Axes,
#     f: Expr,
#     levels: Iterable[float] | int | None = None,
#     use_cache: bool | tuple[bool, bool] = (True, False),
#     **kwargs,
# ) -> None:
#     ...


@singledispatch
def _plot_contours(arg, *_, **__):
    raise MultipleDispatchTypeError(arg, _plot_contours)


@_plot_contours.register(Axes)
def _(
    ax: Axes,
    arg,
    *args,
    **kwargs,
):
    return _plot_contours(arg, ax, *args, **kwargs)


@_plot_contours.register(Function)
@_plot_contours.register(Expr)
def _(
    u: Function | Expr,
    ax: Axes,
    levels: Iterable[float] | int | None = None,
    grid: bool | None = None,
    use_cache: bool | tuple[bool, bool] = (True, False),
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    u_npy = as_npy_function(u, grid, use_cache, mesh)
    return _plot_contours(
        u_npy, ax, levels, grid, **kwargs
    )


@_plot_contours.register(GridFunction)
def _(
    u: GridFunction,
    ax: Axes,
    levels: Iterable[float] | int | None = None,
    grid: bool | None = None,
    **kwargs,
) -> None:
    xyz = (*u.mesh.axes, u.values)
    return _plot_contours(xyz, ax, levels, grid, **kwargs)


@_plot_contours.register(TriFunction)
def _(
    u: TriFunction,
    ax: Axes,
    levels: Iterable[float] | int | None = None,
    grid: bool | None = None,
    **kwargs,
) -> None:
    xyz = (u.mesh.x_coordinates, u.mesh.y_coordinates, u.values)
    return _plot_contours(xyz, ax, levels, grid, u.mesh.triangulation, **kwargs)


@_plot_contours.register(tuple)
def _(  
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    ax: Axes,
    levels: Iterable[float] | int | None,
    grid: bool | None = None,
    triang: Triangulation | None = None,
    **kwargs,
) -> None:
    x, y, z = xyz

    _plt_kwargs = dict(linestyles="solid", color="black", linewidths=LW)
    _axs_kwargs = dict(x_label='$x$', y_label='$y$', aspect='equal')
    _axs_kwargs.update(x_lims=x, y_lims=y)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)
    filter_kwargs(set_axes)(ax, **_kwargs)

    if triang is not None:
        filter_kwargs(ax.tricontour, ContourSet)(triang, z, levels=levels, **_kwargs)
    else:
        if grid is None:
            grid = is_grid((x, y))
        if not grid:
            filter_kwargs(ax.tricontour, ContourSet)(x, y, z, levels=levels, **_kwargs)
        else:
            filter_kwargs(ax.contour, ContourSet)(x, y, z.T, levels=levels, **_kwargs)
            
                

plot_contours = optional_ax(_plot_contours)


@optional_fig_axs
def plot_colormap_multifigure(
    fig: Figure,
    axs_main: list[Axes],
    axs_cbar: list[Axes | None],
    u: Iterable[Function | Expr],
    cmaps: Iterable[str] | str = 'hot',
    titles: Iterable[str] | None = None,
    **plt_kwargs,
) -> None:
    
    if titles is None:
        titles = [None] * len(u)

    if isinstance(cmaps, str):
        cmaps = [cmaps] * len(u)

    for ui, cmap, title, ax_main, ax_cbar in zip(u, cmaps, titles, axs_main, axs_cbar):
        plot_colormap(
            fig, ax_main, ui, 
            title=title, 
            cmap=cmap, 
            colorbar=False, 
            **plt_kwargs,
        ) 
        if isinstance(ax_cbar, tuple):
            cmap: ScalarMappable = ax_main.collections[0]
            cmap.set_clim(*ax_cbar)
            fig.colorbar(cmap, ax=ax_main)
        if ax_cbar is not None:
            fig.colorbar(ax_main.collections[0], ax_cbar)