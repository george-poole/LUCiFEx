import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from dolfinx.mesh import Mesh

from ..utils.fenicsx_utils import (
    mesh_coordinates,
    mesh_axes,
    is_grid,
    is_simplicial,
)
from ..utils.py_utils import filter_kwargs
from ..mesh import as_grid_mesh, as_tri_mesh, as_quad_mesh
from .utils import optional_ax, set_axes, LW


@optional_ax
def plot_mesh(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool = True,
    **kwargs,
) -> None:
    
    _axs_kwargs = dict(x_label="$x$", aspect='equal')
    _plt_kwargs = dict(color='black', linewidth=LW)
    _kwargs = _axs_kwargs | _plt_kwargs
    _kwargs.update(kwargs)

    dim = mesh.geometry.dim
    match dim:
        case 1:
            _plot_interval_mesh(ax, mesh, use_cache, **_kwargs)
        case 2:
            _kwargs.update(y_label="$y$")
            _plot_rectangle_mesh(ax, mesh, use_cache, **_kwargs)
        case 3:
            raise ValueError("3D plotting not supported.")
        case _:
            raise ValueError


def _plot_interval_mesh(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    *,
    y_axis: bool = False,
    **kwargs,
) -> None:
    x, = mesh_axes(use_cache=use_cache)(mesh)

    _kwargs = dict(
        marker="o",
        markersize=5,
        markerfacecolor="black",
        markeredgecolor="black",
    )
    _kwargs.update(kwargs)

    filter_kwargs(set_axes)(
        ax,
        x_lims=x,
        **_kwargs,
    )
    filter_kwargs(ax.plot, Line2D)(x, [0.0] * len(x), **_kwargs)
    if not y_axis:
        ax.yaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))


def _plot_rectangle_mesh(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **kwargs,
) -> tuple[Figure, Axes]:
    grid = is_grid(mesh)
    simplicial = is_simplicial(mesh)

    match simplicial, grid:
        case True, _:
            _plot_triangulation(ax, mesh, use_cache, **kwargs)
        case False, True:
            _plot_grid(ax, mesh, use_cache, **kwargs)
        case False, False:
            _plot_quadrangulation(ax, mesh, use_cache, **kwargs)


def _plot_triangulation(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **kwargs,
) -> None:
    tri_mesh = as_tri_mesh(use_cache=use_cache)(mesh)
    filter_kwargs(set_axes)(
        ax,
        x_lims=tri_mesh.x_coordinates,
        y_lims=tri_mesh.y_coordinates,
        **kwargs,
    )
    filter_kwargs(ax.triplot, Line2D)(tri_mesh.triangulation, **kwargs)


def _plot_quadrangulation(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **kwargs,
) -> None:
    _axs_kwargs = dict(x_label="$x$", y_label="$y$", aspect='equal')
    _poly_kwargs = dict(facecolors='white', edgecolors='black', linewidths=LW)
    _kwargs = _poly_kwargs | _axs_kwargs
    _kwargs.update(kwargs)

    quad_mesh = as_quad_mesh(use_cache=use_cache)(mesh)

    quad_poly = filter_kwargs(quad_mesh.polycollection, Collection)(**_kwargs)

    filter_kwargs(set_axes)(
        ax,
        x_lims=quad_mesh.x_coordinates,
        y_lims=quad_mesh.y_coordinates,
        **_kwargs,
    )
    ax.add_collection(quad_poly)


def _plot_grid(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **kwargs,
) -> None:
    _kwargs = dict(linewidths=LW, colors="black")
    _kwargs.update(kwargs)

    grid = as_grid_mesh(use_cache=use_cache)(mesh)
    x, y = grid.axes
    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))

    filter_kwargs(set_axes)(
        ax,
        x_lims=x, 
        y_lims=y, 
        **_kwargs,
    )
    filter_kwargs(ax.vlines, Collection)(x, *ylim, **_kwargs)
    filter_kwargs(ax.hlines, Collection)(y, *xlim, **_kwargs)
