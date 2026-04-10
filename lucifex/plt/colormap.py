from functools import singledispatch
from typing import overload, Iterable, Callable

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from ufl.core.expr import Expr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet
from matplotlib.collections import Collection 
from matplotlib.tri.triangulation import Triangulation

from ..fem import GridFunction, TriFunction, QuadFunction, as_npy_function
from ..utils.fenicsx_utils import (
    is_scalar, is_grid, IsNotScalarError, IsNotGridOrSimplexMeshError,
)
from ..utils.py_utils import create_kws_filterer, replicate_callable, OverloadTypeError

from .utils import (
    LW, set_axes, optional_ax, set_axes, create_colorbar,
    optional_fig_ax, optional_multifig_ax,
)


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    u: Function | Expr,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    use_cache: bool | tuple[bool, bool] = True,
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
    use_cache: bool | tuple[bool, bool] = True,
    **kwargs,
) -> None:
    ...


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    **kwargs,
) -> None:
    ...


@singledispatch
def _plot_colormap(arg, *_, **__):
    raise OverloadTypeError(arg, _plot_colormap)


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
    use_cache: bool | tuple[bool, bool] = True,
    mesh: Mesh | None = None,
    **kwargs,
):
    if not is_scalar(u):
        raise IsNotScalarError(u)
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
    xyz = (*u.mesh.axes, u.value)
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
    xyz = (u.mesh.x_coordinates, u.mesh.y_coordinates, u.value)
    return _plot_colormap(
        xyz, fig, ax, colorbar, grid, triang=u.mesh.triangulation, **kwargs,
    )


@_plot_colormap.register(QuadFunction)
def _(
    u: QuadFunction,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    cbar_ax: Axes | None = None,
    cax: bool = True,
    cbar_title: str | None = None,
    **kwargs,
):
    if grid:
        raise ValueError('Cannot plot a `QuadFunction` as a grid.')

    _poly_kwargs = dict(cmap='hot', edgecolors='face')
    _axs_kwargs = dict(
        x_lims=u.mesh.x_coordinates, 
        y_lims=u.mesh.y_coordinates, 
        x_label='$x$',
        y_label='$y$', 
        aspect='equal',
    )
    _kwargs = _poly_kwargs | _axs_kwargs
    _kwargs.update(kwargs)
    create_kws_filterer(set_axes)(ax, **_kwargs)

    quad_poly = create_kws_filterer(u.mesh.polycollection, ('array', Collection))(
        array=u.cell_values, 
        **_kwargs,
    )
    create_kws_filterer(quad_poly.set_clim)(**_kwargs)
    ax.add_collection(quad_poly)

    if colorbar is not False:
        limits = None if colorbar is True else colorbar
        create_colorbar(fig, ax, quad_poly, limits, cbar_ax, cax, cbar_title)
    


@_plot_colormap.register(tuple)
def _(
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    grid: bool | None = None,
    triang: Triangulation | None = None,
    cbar_ax: Axes | None = None,
    cax: bool = True,
    cbar_title: str | None = None,
    **kwargs,
):
    x, y, z = xyz

    _plt_kwargs = dict(cmap="hot", shading="gouraud")
    _axs_kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(kwargs)
    create_kws_filterer(set_axes)(ax, **_kwargs)

    if isinstance(colorbar, tuple):
        vmin, vmax = colorbar
        _kwargs.update(vmin=vmin, vmax=vmax)

    if triang is not None:
        cmap = create_kws_filterer(ax.tripcolor)(triang, z, **_kwargs)
    else:
        if grid is None:
            grid = is_grid((x, y))
        if grid:
            cmap = create_kws_filterer(ax.pcolormesh)(x, y, z.T, **_kwargs)
        else:
            cmap = create_kws_filterer(ax.tripcolor, ('triangles', 'mask'))(x, y, z, **_kwargs)

    if colorbar is not False:
        limits = None if colorbar is True else colorbar
        create_colorbar(fig, ax, cmap, limits, cbar_ax, cax, cbar_title)


@replicate_callable(optional_fig_ax(_plot_colormap))
def plot_colormap(): pass


@overload
def _plot_contours(
    ax: Axes,
    u: Function | Expr,
    levels: Iterable[float] | int | None = None,
    grid: bool | None = None,
    use_cache: bool | tuple[bool, bool] = True,
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    ...


@overload
def _plot_contours(
    ax: Axes,
    u: GridFunction | TriFunction,
    levels: Iterable[float] | int | None = None,
    grid: bool | None = None,
    **kwargs,
) -> None:
    ...


@overload
def _plot_contours(
    ax: Axes,
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    levels: Iterable[float] | int | None,
    grid: bool | None = None,
    triang: Triangulation | None = None,
    **kwargs,
) -> None:
    ...


@singledispatch
def _plot_contours(arg, *_, **__):
    raise OverloadTypeError(arg, _plot_contours)


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
    use_cache: bool | tuple[bool, bool] = True,
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
    xyz = (*u.mesh.axes, u.value)
    return _plot_contours(xyz, ax, levels, grid, **kwargs)


@_plot_contours.register(TriFunction)
def _(
    u: TriFunction,
    ax: Axes,
    levels: Iterable[float] | int | None = None,
    grid: bool | None = None,
    **kwargs,
) -> None:
    xyz = (u.mesh.x_coordinates, u.mesh.y_coordinates, u.value)
    return _plot_contours(xyz, ax, levels, grid, u.mesh.triangulation, **kwargs)


@_plot_contours.register(QuadFunction)
def _(
    *_,
    **__,
) -> None:
    raise IsNotGridOrSimplexMeshError


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

    _plt_kwargs = dict(linestyles="solid", linewidths=LW)
    _axs_kwargs = dict(x_label='$x$', y_label='$y$', aspect='equal')
    _axs_kwargs.update(x_lims=x, y_lims=y)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(kwargs)
    create_kws_filterer(set_axes)(ax, **_kwargs)

    if triang is not None:
        create_kws_filterer(ax.tricontour, ContourSet)(triang, z, levels=levels, **_kwargs)
    else:
        if grid is None:
            grid = is_grid((x, y))
        if grid:
            create_kws_filterer(ax.contour, ContourSet)(x, y, z.T, levels=levels, **_kwargs)
        else:
            create_kws_filterer(ax.tricontour, ContourSet)(x, y, z, levels=levels, **_kwargs)
                

@replicate_callable(optional_ax(_plot_contours))
def plot_contours(): pass


@optional_multifig_ax
def plot_colormap_multifigure(
    fig: Figure,
    axs_main: list[Axes],
    axs_cbar: list[Axes] | list[tuple[float, float] | None],
    u: Iterable[Function | GridFunction | Expr],
    cmap: Iterable[str] | str = 'hot',
    title: Iterable[str] | None = None,
    *,
    posthook: Callable[[Figure, Axes], None] | None = None,
    **kwargs,
) -> None:
    
    if title is None:
        title = [None] * len(u)

    if isinstance(cmap, str):
        cmap = [cmap] * len(u)

    vmin, vmax = None, None
    for ax_cb in axs_cbar:
        if isinstance(ax_cb, tuple):
            vmin, vmax = ax_cb
            break

    _kwargs = kwargs.copy()
    if vmin is not None:
        _kwargs.update(vmin=vmin)
    if vmax is not None:
        _kwargs.update(vmax=vmax)

    for ui, cmp, ttl, ax_m, ax_cb in zip(u, cmap, title, axs_main, axs_cbar): 
        if isinstance(ax_cb, tuple):
            _colorbar = ax_cb
            _cbar_ax = ax_m
            _cax = False
        elif isinstance(ax_cb, Axes):
            _colorbar = True
            _cbar_ax = ax_cb
            _cax = True
        else:
            _colorbar = False
            _cbar_ax = None
            _cax = True
        plot_colormap(
            fig, ax_m, ui, 
            title=ttl, 
            cmap=cmp, 
            colorbar=_colorbar,
            cbar_ax=_cbar_ax, 
            cax=_cax,
            **_kwargs,
        )
        if posthook is not None:
            posthook(fig, ax_m)