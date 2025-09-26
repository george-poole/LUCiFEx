import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from dolfinx.mesh import Mesh

from ..mesh.cartesian import CellType
from ..utils import (
    coordinates,
    is_structured,
    quadrangulation,
    triangulation,
)

from ..utils import grid, filter_kwargs, FixMeError
from .utils import optional_ax, set_axes


@optional_ax
def plot_mesh(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    **plt_kwargs,
) -> None:
    dim = mesh.geometry.dim
    match dim:
        case 1:
            _plot_interval_mesh(ax, mesh, title, **plt_kwargs)
        case 2:
            _plot_rectangle_mesh(ax, mesh, title, **plt_kwargs)
        case 3:
            raise ValueError("3D plotting not supported")
        case _:
            raise ValueError


def _plot_interval_mesh(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    **plt_kwargs,
) -> tuple[Figure, Axes]:
    raise FixMeError()


def _plot_rectangle_mesh(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    **plt_kwargs,
) -> tuple[Figure, Axes]:
    cell_type = mesh.topology.cell_name()
    structured = is_structured(mesh)

    match cell_type, structured:
        case CellType.TRIANGLE, True | False:
            _rectangle_triangulation(ax, mesh, title, **plt_kwargs)
        case CellType.QUADRILATERAL, True:
            _rectangle_grid(ax, mesh, title, **plt_kwargs)
        case CellType.QUADRILATERAL, False:
            _rectangle_quadrangulation(ax, mesh, title, **plt_kwargs)
        case _:
            raise ValueError


def _rectangle_triangulation(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable for structured and unstructured meshes"""

    _axs_kwargs = dict(x_label="$x$", y_label="$y$",aspect='equal')
    _plt_kwargs = dict(color='black', linewidth=0.75)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    trigl = triangulation(mesh)
    filter_kwargs(set_axes)(
        ax,
        x_lims=trigl.x,
        y_lims=trigl.y,
        title=title,
        **_kwargs,
    )
    filter_kwargs(ax.triplot, Line2D)(trigl, **_kwargs)


def _rectangle_quadrangulation(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable for structured and unstructured meshes"""

    _axs_kwargs = dict(x_label="$x$", y_label="$y$",aspect='equal')
    _plt_kwargs = dict(facecolor=None, edgecolor='black', linewidth=0.75)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    quadl = quadrangulation(mesh)

    for attr, value in _kwargs.items():
        setter = f'set_{attr}'
        if hasattr(quadl, setter):
            getattr(quadl, setter)(value)

    x_coordinates, y_coordinates = coordinates(mesh)
    filter_kwargs(set_axes)(
        ax,
        x_lims=x_coordinates,
        y_lims=y_coordinates,
        title=title,
        **_kwargs,
    )
    ax.add_collection(quadl)


def _rectangle_grid(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable only for structured meshes"""

    _axs_kwargs = dict(x_label="$x$", y_label="$y$",aspect='equal')
    _plt_kwargs = dict(color='black', linewidth=0.75)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    x, y = grid(use_cache=True)(mesh)
    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))

    filter_kwargs(set_axes)(
        ax,
        x_lims=x, 
        y_lims=y, 
        title=title, 
        **_kwargs,
    )

    ax.vlines(x, *ylim, **_plt_kwargs)
    ax.hlines(y, *xlim, **_plt_kwargs)
