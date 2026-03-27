from typing import Callable, Callable, Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utils.py_utils import create_kws_filterer
from .utils import optional_fig_ax, set_axes, create_colorbar


@optional_fig_ax
def plot_scatter(
    fig: Figure,
    ax: Axes,
    xyz: tuple[Iterable[float], Iterable[float], Iterable[float]],
    size: float | Callable[[Iterable[float]], np.ndarray] = 36.0,
    cmap: str | tuple[str, str] | Iterable[str] = "plasma",
    indicator: Callable[..., bool | str | int] | None = None,
    ax_cbar: Axes | None = None,
    cax: bool = True,
    **kwargs,
) -> None:
    
    xs, ys, z_data = xyz

    if not all(len(i) == len(z_data) for i in (xs, ys)):
        raise ValueError('Arrays must all be the same size.')
    
    _kwargs = dict()
    _kwargs.update(kwargs)
    create_kws_filterer(set_axes)(ax, **_kwargs)


    match cmap:
        case true_marker, false_marker:
            if indicator is None:
                indicator = bool
            i_true = [i for i, z in enumerate(z_data) if indicator(z)]
            i_false = [i for i, z in enumerate(z_data) if not indicator(z)]
            create_kws_filterer(ax.scatter)(
                xs[i_true], ys[i_true], c=z_data[i_true], marker=true_marker, **_kwargs,
            )
            create_kws_filterer(ax.scatter)(
                xs[i_false], ys[i_false], c=z_data[i_false], marker=false_marker, **_kwargs,
            )
        case markers if not isinstance(markers, str):
            if indicator is None:
                raise TypeError('Marker indicator cannot be None.')
            for j, m in enumerate(markers):
                i_marked = [i for i, z in enumerate(z_data) if indicator(z) in (m, j)]
                create_kws_filterer(ax.scatter)(
                    xs[i_marked], ys[i_marked], c=z_data[i_marked], marker=m, **_kwargs,
                )
        case cmap:
            if callable(size):
                size = size(z_data)
            path_clc = create_kws_filterer(ax.scatter)(
                xs, ys, c=z_data, cmap=cmap, s=size, **_kwargs
            )
            create_colorbar(fig, ax, path_clc, limits=None, ax_cbar=ax_cbar, cax=cax)


def scatter_size(
    z_data: Iterable[float], 
    s_min: float = 10.0, 
    s_max: float = 400.0, 
    n: int = 1,
) -> np.ndarray:
    """
    Defining a marker size based on the point's z-value

    s(z) = m·zⁿ + c
    """
    z_data = np.asarray(z_data)
    m = (s_max - s_min) / (np.max(z_data) ** n - np.min(z_data) ** n)
    c = s_max - m * np.max(z_data) ** n
    return m * z_data + c
