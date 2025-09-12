import numpy as np
from dolfinx.fem import Function
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utils import is_vector, grid, fem_function_components, is_structured
from .utils import set_axes, optional_ax


@optional_ax
def plot_quiver(
    ax: Axes,
    f: Function,
    title: str | None = None,
    n_arrow: tuple[int, int] = (30, 30),
    axis_names: tuple[str, str] = ("x", "y"),
) -> tuple[Figure, Axes]:
    """Creates a quiver plot of a vector-valued 2D function

    `f(x,y) = (fx(x,y), fy(x,y))`
    """

    if not is_vector(f, dim=2):
        raise ValueError(
            "Quiver plots are for vector-valued functions of dimension 2 only."
        )

    if not is_structured(f.function_space.mesh):
        raise NotImplementedError
    
    fx, fy = fem_function_components(('P', 1), f)

    x, y = grid(use_cache=True)(f.function_space.mesh)
    fx_np = grid(fx)
    fy_np = grid(fy)
    set_axes(ax, x, y, title, *axis_names)
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
    f: Function,
    title: str | None = None,
    density: float = 1.0,
    color: str = 'black',
    axis_names: tuple[str, str] = ("x", "y"),
):
    """Creates a streamline plot of a vector-valued 2D function

    `f(x,y) = (fx(x,y), fy(x,y))`
    """

    if not is_vector(f, dim=2):
        raise ValueError(
            "Streamline plots are for vector-valued functions of dimension 2 only."
        )

    if not is_structured(f.function_space.mesh):
        raise NotImplementedError
    
    fx, fy = fem_function_components(('P', 1), f)

    x, y = grid(use_cache=True)(f.function_space.mesh)
    fx_grid = grid(fx)
    fy_grid = grid(fy)
    set_axes(ax, x, y, title, *axis_names, tex=True)

    if color in list(mpl_colormaps):
        # line colour varies according to vector magnitude |f(x,y)|
        norm = np.sqrt((fx_grid.T) ** 2 + (fy_grid.T) ** 2)
        ax.streamplot(
            x, y, fx_grid.T, fy_grid.T, density=density, color=norm, cmap=color
        )
    else:
        # line colour is fixed
        ax.streamplot(x, y, fx_grid.T, fy_grid.T, density=density, color=color)
