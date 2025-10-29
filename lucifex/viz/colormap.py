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
from ..utils import (is_scalar, grid, triangulation, finite_element_function, is_cartesian, 
                     filter_kwargs, MultipleDispatchTypeError, UnstructuredQuadError, extract_mesh)

from .utils import LW, set_axes, optional_ax, set_axes, optional_fig_ax


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    f: Function,
    colorbar: bool | tuple[float, float] = True,
    cartesian: bool | None = None,
    use_cache: bool = False,
    **kwargs,
) -> None:
    """Plots colormap of a scalar-valued function"""
    ...


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    expr: Expr,
    colorbar: bool | tuple[float, float] = True,
    cartesian: bool | None = None,
    use_cache: bool = False,
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    """Plots colormap of a scalar-valued expression"""
    ...


@overload
def _plot_colormap(
    fig: Figure,
    ax: Axes, 
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Triangulation, np.ndarray],
    colorbar: bool | tuple[float, float] = True,
    cartesian: bool | None = None,
    **kwargs,
) -> None:
    ...


@singledispatch
def _plot_colormap(f, *_, **__):
    raise MultipleDispatchTypeError(f, _plot_colormap)


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
def _(
    f: Function,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    cartesian: bool | None = None,
    use_cache: tuple[bool, bool] = (True, False),
    **kwargs,
):
    if not is_scalar(f):
        raise ValueError("Colormap plots must be of scalar-valued quantities.")

    mesh = f.function_space.mesh
    cell_type = mesh.topology.cell_name()

    if cartesian is None:
        cartesian = is_cartesian(use_cache=use_cache[0])(mesh)

    match cell_type, cartesian:
        case CellType.TRIANGLE, False:
            trigl = triangulation(use_cache=use_cache[0])(f.function_space.mesh)
            f_tri = triangulation(use_cache=use_cache[1])(f)
            xyz = (trigl, f_tri)
        case CellType.TRIANGLE | CellType.QUADRILATERAL, True:
            x, y = grid(use_cache=use_cache[0])(f.function_space.mesh)
            f_grid = grid(use_cache=use_cache[1])(f)
            xyz = (x, y, f_grid)
        case CellType.QUADRILATERAL, False:
            raise UnstructuredQuadError
        case _:
            raise ValueError(f'Unexpected cell type {cell_type}')

    return _plot_colormap(
        xyz, fig, ax, colorbar, cartesian, **kwargs
    )


@_plot_colormap.register(Expr)
def _(
    expr: Expr,
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    cartesian: bool | None = None,
    use_cache: bool = False,
    mesh: Mesh | None = None,
    **kwargs,
):
    if mesh is None:
        mesh = extract_mesh(expr)
    f = finite_element_function((mesh, 'P', 1), expr)
    return _plot_colormap(
        f,
        fig, 
        ax,
        colorbar,
        cartesian,
        use_cache,
        **kwargs
    )


@_plot_colormap.register(tuple)
def _(
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Triangulation, np.ndarray],
    fig: Figure,
    ax: Axes,
    colorbar: bool | tuple[float, float] = True,
    cartesian: bool | None = None,
    **kwargs,
):
    triang_available = len(xyz) == 2

    if triang_available:
        tri, z = xyz
        assert isinstance(tri, Triangulation)
        x, y = tri.x, tri.y
    else:
        x, y, z = xyz

    _plt_kwargs = dict(cmap="hot", shading="gouraud")
    _axs_kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    filter_kwargs(set_axes)(ax, **_kwargs)

    if triang_available:
        cmap = filter_kwargs(ax.tripcolor)(tri, z, **_kwargs)
    else:
        if cartesian is None:
            cartesian = bool(len(x) == len(np.unique(x)) and len(y) == len(np.unique(y)))
        if not cartesian:
            cmap = filter_kwargs(ax.tripcolor, ('triangles', 'mask'))(x, y, z, **_kwargs)
        else:
            cmap = filter_kwargs(ax.pcolormesh)(x, y, z.T, **_kwargs)

    if colorbar is not False:
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(cmap, ax_cbar)
        if isinstance(colorbar, tuple):
            assert len(colorbar) == 2
            cmap.set_clim(*colorbar)


plot_colormap = optional_fig_ax(_plot_colormap)


@optional_ax
def plot_contours(
    ax: Axes,
    f: Function | tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Triangulation, np.ndarray],
    levels: Iterable[float] | int | None = None,
    use_cache: tuple[bool, bool] = (True, False),
    **kwargs,
) -> None:
    """Plots contours of a scalar-valued function"""
    return _plot_contours(ax, f, levels, use_cache, **kwargs)


def _plot_contours(
    ax: Axes,
    f: Function | tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Triangulation, np.ndarray],
    levels: Iterable[float] | int | None,
    use_cache: tuple[bool, bool],
    **kwargs,
) -> None:
    if isinstance(f, tuple):
        triang_available = len(f) == 2
        
        if triang_available:
            tri, z = f
            assert isinstance(tri, Triangulation)
            x, y = tri.x, tri.y
        else:
            x, y, z = f

        _plt_kwargs = dict(linestyles="solid", color="black", linewidths=LW)
        _axs_kwargs = dict(x_label='$x$', y_label='$y$', aspect='equal')
        _axs_kwargs.update(x_lims=x, y_lims=y)
        _kwargs = _plt_kwargs | _axs_kwargs
        _kwargs.update(**kwargs)
        filter_kwargs(set_axes)(ax, **_kwargs)

        if triang_available:
            filter_kwargs(ax.tricontour, ContourSet)(tri, z, levels=levels, **_kwargs)
        else:
            filter_kwargs(ax.contour, ContourSet)(x, y, z.T, levels=levels, **_kwargs)

    else:
        cartesian = is_cartesian(use_cache=use_cache[0])(f.function_space.mesh)
        if not cartesian:
            trigl = triangulation(use_cache=use_cache[0])(f.function_space.mesh)
            f_trigl = triangulation(use_cache=use_cache[1])(f)
            return _plot_contours(ax, (trigl, f_trigl), levels, **kwargs)
        else:
            x, y = grid(use_cache=use_cache[0])(f.function_space.mesh)
            f_grid = grid(use_cache=use_cache[1])(f)
            return _plot_contours(ax, (x, y, f_grid), levels, use_cache, **kwargs)
        
