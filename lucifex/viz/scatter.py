from typing import Callable, Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import filter_kwargs
from .utils import optional_fig_ax, set_axes


@optional_fig_ax
def plot_scatter(
    fig: Figure,
    ax: Axes,
    xyz_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    cmap: str | tuple[str, str] = "viridis",
    s: float | Callable[[np.ndarray], np.ndarray] = 36.0,
    **kwargs,
) -> None:
    
    _kwargs = dict()
    _kwargs.update(kwargs)

    filter_kwargs(set_axes)(ax, **_kwargs)

    x_data, y_data, z_data = xyz_data

    match cmap:
        case true_marker, false_marker:
            i_true = [i for i, j in enumerate(z_data) if bool(j) is True]
            i_false = [i for i, j in enumerate(z_data) if bool(j) is False]
            filter_kwargs(ax.scatter)(x_data[i_true], y_data[i_true], c=z_data[i_true], marker=true_marker, **_kwargs)
            filter_kwargs(ax.scatter)(x_data[i_false], y_data[i_false], c=z_data[i_false], marker=false_marker, **_kwargs)
        case cmap:
            if callable(s):
                s = s(z_data)
            ax_scatter = filter_kwargs(ax.scatter)(
                x_data, y_data, c=z_data, cmap=cmap, s=s, **_kwargs
            )
            divider = make_axes_locatable(ax)
            colorbar_ax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(ax_scatter, colorbar_ax)


def marker_size_func(
    z_data: np.ndarray, s_min: float = 10.0, s_max: float = 400.0, n: int = 1,
) -> np.ndarray:
    """
    Defining a marker size based on the point's z-value

    s(z) = m·zⁿ + c
    """
    m = (s_max - s_min) / (np.max(z_data) ** n - np.min(z_data) ** n)
    c = s_max - m * np.max(z_data) ** n
    return m * z_data + c
