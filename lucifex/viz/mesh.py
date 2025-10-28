import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from dolfinx.mesh import Mesh

from ..mesh.cartesian import CellType
from ..utils import (
    mesh_coordinates,
    is_cartesian,
    quadrangulation,
    triangulation,
    grid, 
    filter_kwargs, 
    ToDoError,
)
from .utils import optional_ax, set_axes


@optional_ax
def plot_mesh(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool = True,
    **plt_kwargs,
) -> None:
    dim = mesh.geometry.dim
    match dim:
        case 1:
            _plot_interval_mesh(ax, mesh, use_cache, **plt_kwargs)
        case 2:
            _plot_rectangle_mesh(ax, mesh, use_cache, **plt_kwargs)
        case 3:
            raise ValueError("3D plotting not supported.")
        case _:
            raise ValueError


def _plot_interval_mesh(
    ax: Axes, 
    mesh: Mesh,
    **plt_kwargs,
) -> tuple[Figure, Axes]:
    raise ToDoError


def _plot_rectangle_mesh(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **plt_kwargs,
) -> tuple[Figure, Axes]:
    cell_type = mesh.topology.cell_name()
    cartesian = is_cartesian(mesh)

    match cell_type, cartesian:
        case CellType.TRIANGLE, True | False:
            _plot_triangulation(ax, mesh, use_cache, **plt_kwargs)
        case CellType.QUADRILATERAL, True:
            _plot_grid(ax, mesh, use_cache, **plt_kwargs)
        case CellType.QUADRILATERAL, False:
            _plot_quadrangulation(ax, mesh, use_cache, **plt_kwargs)
        case _:
            raise ValueError


def _plot_triangulation(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable for Cartesian and unstructured meshes"""

    _axs_kwargs = dict(x_label="$x$", y_label="$y$",aspect='equal')
    _plt_kwargs = dict(color='black', linewidth=0.75)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    trigl = triangulation(use_cache=use_cache)(mesh)
    filter_kwargs(set_axes)(
        ax,
        x_lims=trigl.x,
        y_lims=trigl.y,
        **_kwargs,
    )
    filter_kwargs(ax.triplot, Line2D)(trigl, **_kwargs)


def _plot_quadrangulation(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable for Cartesian and unstructured meshes"""

    _axs_kwargs = dict(x_label="$x$", y_label="$y$",aspect='equal')
    _poly_kwargs = dict(facecolor='white', edgecolor='black', linewidth=0.75)
    _kwargs = _poly_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    quadl = quadrangulation(use_cache=use_cache)(mesh)
    quadl = filter_kwargs(quadrangulation, Collection)(mesh, **_kwargs)

    for attr, value in _kwargs.items():
        setter = f'set_{attr}'
        if hasattr(quadl, setter):
            getattr(quadl, setter)(value)

    x_coordinates, y_coordinates = mesh_coordinates(mesh)
    filter_kwargs(set_axes)(
        ax,
        x_lims=x_coordinates,
        y_lims=y_coordinates,
        **_kwargs,
    )
    ax.add_collection(quadl)


def _plot_grid(
    ax: Axes, 
    mesh: Mesh,
    use_cache: bool,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable only for Cartesian meshes"""

    _axs_kwargs = dict(x_label="$x$", y_label="$y$",aspect='equal')
    _plt_kwargs = dict(color='black', linewidth=0.75)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    x, y = grid(use_cache=use_cache)(mesh)
    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))

    filter_kwargs(set_axes)(
        ax,
        x_lims=x, 
        y_lims=y, 
        **_kwargs,
    )

    ax.vlines(x, *ylim, **_plt_kwargs)
    ax.hlines(y, *xlim, **_plt_kwargs)
