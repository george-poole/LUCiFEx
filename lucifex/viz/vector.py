from typing import Callable

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver

from ..utils import (
    is_vector, grid, triangulation, create_fem_function, extract_mesh, ShapeError, 
    NonCartesianQuadMeshError, is_simplicial, get_component_fem_functions, 
    is_cartesian, filter_kwargs,
)
from .utils import set_axes, optional_ax


@optional_ax
def plot_quiver(
    ax: Axes,
    f: Function | tuple[Function, Function] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Expr,
    n_arrow: int | tuple[int, int] = 1,
    use_cache: tuple[bool, bool] = (True, False),
    mesh: Mesh | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plots quiver arrows of a two-dimensional vector
    """
    if isinstance(f, tuple) and len(f) == 4:
        return _plot_quiver(ax, f, n_arrow, **kwargs)

    fx, fy = _xy_components(f, mesh)
    x, y, fx_np, fy_np = _x_y_fx_fy_arrays(fx, fy, use_cache, 'Quiver plotting')
    
    return _plot_quiver(ax, (x, y, fx_np, fy_np), n_arrow, **kwargs)


def _plot_quiver(
    ax: Axes,
    f: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_arrow: int | tuple[int, int],
    arrow_slc,
    **kwargs,
) -> tuple[Figure, Axes]:
    
    x, y, fx, fy = f
    _kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _kwargs.update(**kwargs)
    filter_kwargs(set_axes)(ax, **_kwargs)

    if len(np.shape(x)) ==  2:
        tri = True
    else:
        tri = False

    if tri:
        n_freq = int(np.ceil(len(x) / n_arrow))
        x_quiv = x[::n_freq]
        y_quiv = y[::n_freq]
        fx_quiv = fx[::n_freq]
        fy_quiv = fy[::n_freq]
    else:
        if isinstance(n_arrow, int):
            n_arrow = (n_arrow, n_arrow)
        nx_arrow, ny_arrow = n_arrow
        nx_freq = int(np.ceil(len(x) / nx_arrow))
        ny_freq = int(np.ceil(len(y) / ny_arrow))
        x_quiv = x[::nx_freq]
        y_quiv = x[::nx_freq]
        fx_quiv = fx[::nx_freq, ::ny_freq].T,
        fy_quiv = fy[::nx_freq, ::ny_freq].T,

    filter_kwargs(ax.quiver, Quiver)(
        x_quiv,
        y_quiv,
        fx_quiv,
        fy_quiv,
        **_kwargs,
    )


@optional_ax
def plot_streamlines(
    ax: Axes,
    f: Function | tuple[Function, Function] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Expr,
    density: float = 1.0,
    color: str | tuple[str, Callable]= 'black',
    use_cache: tuple[bool, bool] = (True, False),
    mesh: Mesh | None = None,
    **kwargs,
):
    """
    Plots streamlines of a two-dimensional vector
    """
    if isinstance(f, tuple) and len(f) == 4:
        return _plot_streamlines(ax, f, density, color, **kwargs)
    
    fx, fy = _xy_components(f, mesh)
    x, y, fx_np, fy_np = _x_y_fx_fy_arrays(fx, fy, use_cache, 'Streamline plotting')

    return _plot_streamlines(ax, (x, y, fx_np, fy_np), density, color, **kwargs)


def _plot_streamlines(
    ax: Axes,
    f: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    density,
    color: str | tuple[str, Callable],
    **kwargs,
):
    x, y, fx, fy = f

    if isinstance(color, str):
        color_func = lambda fx, fy: np.sqrt((fx) ** 2 + (fy) ** 2)
    else:
        color, color_func = color

    _axs_kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _axs_kwargs.update(**kwargs)
    filter_kwargs(set_axes)(ax, **_axs_kwargs)

    if color in list(mpl_colormaps):
        # line colour varying according to vector magnitude |f(x,y)|
        norm = color_func(fx.T, f.T)
        ax.streamplot(
            x, y, fx.T, fy.T, density=density, color=norm, cmap=color
        )
    else:
        # fixed line colour
        ax.streamplot(x, y, fx.T, fy.T, density=density, color=color)


def _xy_components(
    f: Function | tuple[Function, Function] | Expr,
    mesh: Mesh | None,
) -> tuple[Function, Function]:
    if isinstance(f, Expr) and not isinstance(f, Function):
        if mesh is None:
            mesh = extract_mesh(f)
        f = create_fem_function((mesh, 'P', 1, 2), f)
    
    if isinstance(f, Function):
        if not is_vector(f, dim=2):
            raise ShapeError(f, (2, ))
        
        fx, fy = get_component_fem_functions(('P', 1), f)
    else:
        fx, fy = (create_fem_function(('P', 1), i, try_identity=True) for i in f)

    return fx, fy


def _x_y_fx_fy_arrays(
    fx: Function, 
    fy: Function,
    use_cache: tuple[bool, bool],
    error_msg: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh = fx.function_space.mesh
    use_mesh_cache = use_func_cache = use_cache
    if is_cartesian(mesh):
        x, y = grid(use_cache=use_mesh_cache)(mesh)
        fx_np = grid(use_cache=use_func_cache)(fx)
        fy_np = grid(use_cache=use_func_cache)(fy)
    else:
        if not is_simplicial(mesh):
            raise NonCartesianQuadMeshError(error_msg)
        trigl = triangulation(use_cache=use_mesh_cache)(mesh)
        triangles = trigl.triangles
        x = trigl.x[triangles]
        y = trigl.y[triangles]
        fx_np = triangulation(use_cache=use_func_cache)(fx)[triangles]
        fy_np = triangulation(use_cache=use_func_cache)(fy)[triangles]

    return x, y, fx_np, fy_np

