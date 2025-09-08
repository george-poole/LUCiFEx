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

from ..utils import grid, filter_kwargs
from .utils import optional_ax, set_axes


@optional_ax
def plot_mesh(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    vertices: bool = False,
    **plt_kwargs,
) -> None:
    dim = mesh.geometry.dim
    match dim:
        case 1:
            _plot_interval_mesh(ax, mesh, title, vertices, **plt_kwargs)
        case 2:
            _plot_rectangle_mesh(ax, mesh, title, vertices, **plt_kwargs)
        case 3:
            raise ValueError("3D plotting not supported")
        case _:
            raise ValueError


def _plot_interval_mesh(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    vertices: bool = True,
    **plt_kwargs,
) -> tuple[Figure, Axes]:
    raise NotImplementedError # TODO


def _plot_rectangle_mesh(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    vertices: bool = False,
    **plt_kwargs,
) -> tuple[Figure, Axes]:
    cell_type = mesh.topology.cell_name()
    structured = is_structured(mesh)

    match cell_type, structured:
        case CellType.TRIANGLE, True | False:
            _rectangle_triangulation(ax, mesh, title, vertices, **plt_kwargs)
        case CellType.QUADRILATERAL, True:
            _rectangle_grid(ax, mesh, title, vertices, **plt_kwargs)
        case CellType.QUADRILATERAL, False:
            _rectangle_quadrangulation(ax, mesh, title, vertices, **plt_kwargs)
        case _:
            raise ValueError


def _rectangle_triangulation(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    vertices: bool = False,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable for structured and unstructured meshes"""

    _axs_kwargs = dict(x_label="x", y_label="y",aspect='equal', tex=True)
    _plt_kwargs = dict(color='black', linewidth=0.75)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    trigl = triangulation(mesh)
    filter_kwargs(set_axes)(
        ax,
        x_axis=trigl.x,
        y_axis=trigl.y,
        title=title,
        **_kwargs,
    )
    filter_kwargs(ax.triplot, Line2D)(trigl, **_kwargs)

    if vertices:
        _plot_vertices(ax, trigl.x, trigl.y, **_kwargs)


def _rectangle_quadrangulation(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    vertices: bool = False,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable for structured and unstructured meshes"""

    _axs_kwargs = dict(x_label="x", y_label="y",aspect='equal', tex=True)
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
        x_axis=x_coordinates,
        y_axis=y_coordinates,
        title=title,
        **_kwargs,
    )
    ax.add_collection(quadl)

    if vertices:
        _plot_vertices(ax, x_coordinates, y_coordinates, **_kwargs)


def _rectangle_grid(
    ax: Axes, 
    mesh: Mesh,
    title: str | None = None,
    vertices: bool = False,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Suitable only for structured meshes"""

    _axs_kwargs = dict(x_label="x", y_label="y",aspect='equal', tex=True)
    _plt_kwargs = dict(color='black', linewidth=0.75)
    _kwargs = _plt_kwargs | _axs_kwargs
    _kwargs.update(**kwargs)

    x, y = grid(use_cache=True)(mesh)
    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))

    filter_kwargs(set_axes)(
        ax,
        x_axis=x, 
        y_axis=y, 
        title=title, 
        **_kwargs,
    )

    ax.vlines(x, *ylim, **_plt_kwargs)
    ax.hlines(y, *xlim, **_plt_kwargs)

    if vertices:
        x_grid, y_grid = np.meshgrid(x, y)
        _plot_vertices(ax, x_grid, y_grid, **_kwargs)


def _plot_vertices(
    ax: Axes, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """Adds a scatter plot of mesh vertices"""
    filter_kwargs(ax.scatter, Collection)(x, y, s=2, marker="o", **kwargs)
