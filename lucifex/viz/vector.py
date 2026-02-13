from typing import Callable

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.quiver import Quiver

from ..mesh import as_grid_mesh
from ..fem import as_grid_function, TriFunction, GridFunction, as_npy_function
from ..utils.fenicsx_utils import (
    is_vector, create_function, extract_mesh, ShapeError, 
    NonCartesianQuadMeshError, is_simplicial, get_component_functions, 
    is_grid,
)
from ..utils.py_utils import filter_kwargs
from .utils import set_axes, optional_ax


@optional_ax
def plot_quiver(
    ax: Axes,
    f: Function | tuple[Function, Function] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Expr,
    arrow_slc: int | tuple[int, int] = 1,
    use_cache: tuple[bool, bool] = (True, False),
    mesh: Mesh | None = None,
    **kwargs,
) -> None:
    """
    Plots quiver arrows of a two-dimensional vector
    """
    if isinstance(f, tuple) and len(f) == 4:
        return _plot_quiver(ax, f, arrow_slc, **kwargs)

    fx, fy = _xy_components(f, mesh)
    x, y, fx_np, fy_np = _x_y_fx_fy_arrays(fx, fy, use_cache, 'Quiver plotting')
    
    return _plot_quiver(ax, (x, y, fx_np, fy_np), arrow_slc, **kwargs)


def _plot_quiver(
    ax: Axes,
    f: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    arrow_slc: int | tuple[int, int],
    **kwargs,
) -> None:
    
    x, y, fx, fy = f
    _kwargs = dict(x_lims=x, y_lims=y, x_label='$x$', y_label='$y$', aspect='equal')
    _kwargs.update(**kwargs)
    filter_kwargs(set_axes)(ax, **_kwargs)

    if len(np.shape(x)) ==  2:
        tri = True
    else:
        tri = False

    if tri:
        x_quiv = x[::arrow_slc]
        y_quiv = y[::arrow_slc]
        fx_quiv = fx[::arrow_slc]
        fy_quiv = fy[::arrow_slc]
    else:
        if isinstance(arrow_slc, int):
            arrow_slc = (arrow_slc, arrow_slc)
        x_arrow_slc, y_arrow_slc = arrow_slc
        x_quiv = x[::x_arrow_slc]
        y_quiv = y[::y_arrow_slc]
        fx_quiv = fx[::x_arrow_slc, ::y_arrow_slc].T
        fy_quiv = fy[::x_arrow_slc, ::y_arrow_slc].T

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
) -> None:
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
) -> None:
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
        f = create_function((mesh, 'P', 1, 2), f)
    
    if isinstance(f, Function):
        if not is_vector(f, dim=2):
            raise ShapeError(f, (2, ))
        
        fx, fy = get_component_functions(('P', 1), f)
    else:
        fx, fy = (create_function(('P', 1), i, try_identity=True) for i in f)

    return fx, fy


def _x_y_fx_fy_arrays(
    fx: Function, 
    fy: Function,
    use_cache: tuple[bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    fx_np = as_npy_function(fx, use_cache=use_cache)
    fy_np = as_npy_function(fy, use_cache=use_cache)
    
    if isinstance(fx_np, TriFunction):
        triangles = fx_np.mesh.triangles
        x = fx_np.mesh.x_coordinates[triangles]
        y = fx_np.mesh.y_coordinates[triangles]
        fx_new = fx_np.values[triangles]
        fy_new = fy_np.values[triangles]
    
    if isinstance(fx_np, GridFunction):
        x, y = fx_np.mesh.axes
        fx_new = fx_np.values
        fy_new = fy_np.values

    return x, y, fx_new, fy_new

