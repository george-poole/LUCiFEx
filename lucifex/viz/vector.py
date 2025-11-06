from typing import Callable

import numpy as np
from dolfinx.fem import Function
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utils import is_vector, grid, create_function, get_component_functions, is_cartesian
from .utils import set_axes, optional_ax


@optional_ax
def plot_quiver(
    ax: Axes,
    f: Function | tuple[Function, Function] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_arrow: tuple[int, int] = (30, 30),
    use_cache: tuple[bool, bool] = (True, False),
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plots quiver arrows of a vector-valued function (2D)
    """
    if isinstance(f, tuple) and len(f) == 4:
        return _plot_quiver(ax, f, n_arrow, **kwargs)
    
    if isinstance(f, Function):
        if not is_vector(f, dim=2):
            raise ValueError(
                "Quiver plots must be of 2D vector-valued quantities."
            )
        fx, fy = get_component_functions(('P', 1), f)
    else:
        fx, fy = (create_function(('P', 1), i) for i in f)

    if not is_cartesian(fx.function_space.mesh):
        raise ValueError("Quiver plots on unstructured meshes are not supported.")
    
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

    set_axes(ax, x, y, **kwargs)

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
    f: Function | tuple[Function, Function] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    density: float = 1.0,
    color: str | tuple[str, Callable]= 'black',
    use_cache: tuple[bool, bool] = (True, False),
    **kwargs,
):
    """
    Plots streamlines plot of a vector-valued function (2D)
    """
    if isinstance(f, tuple) and len(f) == 4:
        return _plot_streamlines(ax, f, density, color, **kwargs)
    
    if isinstance(f, Function):
        if not is_vector(f, dim=2):
            raise ValueError(
                "Streamline plots must be of 2D vector-valued quantities."
            )
        fx, fy = get_component_functions(('P', 1), f)
    else:
        fx, fy = (create_function(('P', 1), i, try_identity=True) for i in f)

    if not is_cartesian(fx.function_space.mesh):
        raise ValueError("Streamline plots on unstructured meshes are not supported.")

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

    set_axes(ax, x, y, **kwargs)

    if color in list(mpl_colormaps):
        # line colour varying according to vector magnitude |f(x,y)|
        norm = color_func(fx.T, f.T)
        ax.streamplot(
            x, y, fx.T, fy.T, density=density, color=norm, cmap=color
        )
    else:
        # fixed line colour
        ax.streamplot(x, y, fx.T, fy.T, density=density, color=color)
