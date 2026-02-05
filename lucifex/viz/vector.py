from typing import Callable

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utils import (
    is_vector, grid, create_fem_function, extract_mesh, ShapeError, NonCartesianMeshError,
    get_component_fem_functions, is_cartesian, filter_kwargs,
)
from .utils import set_axes, optional_ax


@optional_ax
def plot_quiver(
    ax: Axes,
    f: Function | tuple[Function, Function] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Expr,
    n_arrow: tuple[int, int] = (30, 30),
    use_cache: tuple[bool, bool] = (True, False),
    mesh: Mesh | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plots quiver arrows of a vector-valued function (2D)
    """
    if isinstance(f, Expr) and not isinstance(f, Function):
        if mesh is None:
            mesh = extract_mesh(f)
        f = create_fem_function((mesh, 'P', 1, 2), f)

    if isinstance(f, tuple) and len(f) == 4:
        return _plot_quiver(ax, f, n_arrow, **kwargs)
    
    if isinstance(f, Function):
        if not is_vector(f, dim=2):
            raise ShapeError(f, (2, ))
            
        fx, fy = get_component_fem_functions(('P', 1), f)
    else:
        fx, fy = (create_fem_function(('P', 1), i) for i in f)

    if not is_cartesian(fx.function_space.mesh):
        raise NonCartesianMeshError('Quiver plotting')
    
    use_mesh_cache = use_func_cache = use_cache
    x, y = grid(use_cache=use_mesh_cache)(fx.function_space.mesh)
    fx_grid = grid(use_cache=use_func_cache)(fx)
    fy_grid = grid(use_cache=use_func_cache)(fy)

    return _plot_quiver(ax, (x, y, fx_grid, fy_grid), n_arrow, **kwargs)


def _plot_quiver(
    ax: Axes,
    f: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_arrow: tuple[int, int] = (30, 30),
    **kwargs,
) -> tuple[Figure, Axes]:
    x, y, fx, fy = f

    _axs_kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _axs_kwargs.update(**kwargs)
    filter_kwargs(set_axes)(ax, **_axs_kwargs)

    nx_arrow, ny_arrow = n_arrow
    nx_freq = int(np.ceil(len(x) / nx_arrow))
    ny_freq = int(np.ceil(len(y) / ny_arrow))
    ax.quiver(
        x[::nx_freq],
        y[::ny_freq],
        fx[::nx_freq, ::ny_freq].T,
        fy[::nx_freq, ::ny_freq].T,
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
    Plots streamlines plot of a vector-valued function (2D)
    """
    if isinstance(f, Expr) and not isinstance(f, Function):
        if mesh is None:
            mesh = extract_mesh(f)
        f = create_fem_function((mesh, 'P', 1, 2), f)

    if isinstance(f, tuple) and len(f) == 4:
        return _plot_streamlines(ax, f, density, color, **kwargs)
    
    if isinstance(f, Function):
        if not is_vector(f, dim=2):
            raise ShapeError(f, (2, ))
        
        fx, fy = get_component_fem_functions(('P', 1), f)
    else:
        fx, fy = (create_fem_function(('P', 1), i, try_identity=True) for i in f)

    if not is_cartesian(fx.function_space.mesh):
        raise NonCartesianMeshError('Streamline plotting')

    use_mesh_cache = use_func_cache = use_cache
    x, y = grid(use_cache=use_mesh_cache)(fx.function_space.mesh)
    fx_grid = grid(use_cache=use_func_cache)(fx)
    fy_grid = grid(use_cache=use_func_cache)(fy)

    return _plot_streamlines(ax, (x, y, fx_grid, fy_grid), density, color, **kwargs)


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
