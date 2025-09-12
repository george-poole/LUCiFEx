from typing import Callable

import numpy as np
from dolfinx.fem import Function
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utils import is_vector, grid, fem_function, fem_function_components, is_structured
from .utils import set_axes, optional_ax


@optional_ax
def plot_quiver(
    ax: Axes,
    f: Function | tuple[Function, Function],
    n_arrow: tuple[int, int] = (30, 30),
    **kwargs,
) -> tuple[Figure, Axes]:
    """Creates a quiver plot of a vector-valued 2D function

    `f(x,y) = (fx(x,y), fy(x,y))`
    """
    
    if isinstance(f, tuple):
        fx, fy = (fem_function(('P', 1), i) for i in f)
    else:
        if not is_vector(f, dim=2):
            raise ValueError(
                "Quiver plots are for vector-valued functions of dimension 2 only."
            )
        fx, fy = fem_function_components(('P', 1), f)

    if not is_structured(fx.function_space.mesh):
        raise NotImplementedError

    x, y = grid(use_cache=True)(fx.function_space.mesh)
    fx_np = grid(fx)
    fy_np = grid(fy)
    set_axes(ax, x, y, **kwargs)
    nx_arrow, ny_arrow = n_arrow
    nx_freq = int(np.ceil(len(x) / nx_arrow))
    ny_freq = int(np.ceil(len(y) / ny_arrow))
    ax.quiver(
        x[::nx_freq],
        y[::ny_freq],
        fx_np[::nx_freq, ::ny_freq].T,
        fy_np[::nx_freq, ::ny_freq].T,
    )


@optional_ax
def plot_streamlines(
    ax: Axes,
    f: Function | tuple[Function, Function],
    density: float = 1.0,
    color: str | tuple[str, Callable]= 'black',
    **kwargs,
):
    """Creates a streamline plot of a vector-valued 2D function

    `f(x,y) = (fx(x,y), fy(x,y))`
    """
    
    if isinstance(f, tuple):
        fx, fy = (fem_function(('P', 1), i) for i in f)
    else:
        if not is_vector(f, dim=2):
            raise ValueError(
                "Streamline plots are for vector-valued functions of dimension 2 only."
            )
        fx, fy = fem_function_components(('P', 1), f)

    if not is_structured(fx.function_space.mesh):
        raise NotImplementedError

    x, y = grid(use_cache=True)(fx.function_space.mesh)
    fx_grid = grid(fx)
    fy_grid = grid(fy)
    set_axes(ax, x, y, **kwargs)

    if isinstance(color, str):
        color_func = lambda fx, fy: np.sqrt((fx) ** 2 + (fy) ** 2)
    else:
        color, color_func = color

    if color in list(mpl_colormaps):
        # line colour varying according to vector magnitude |f(x,y)|
        norm = color_func(fx_grid.T, fy_grid.T)
        ax.streamplot(
            x, y, fx_grid.T, fy_grid.T, density=density, color=norm, cmap=color
        )
    else:
        # fixed line colour
        ax.streamplot(x, y, fx_grid.T, fy_grid.T, density=density, color=color)
