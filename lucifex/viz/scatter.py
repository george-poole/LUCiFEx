from typing import Callable, Literal, Callable, Iterable
from types import EllipsisType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import Cycler, cycler as create_cycler

from ..utils import filter_kwargs
from .utils import COLORS, LW, set_legend, optional_fig_ax, set_axes


MARKERS = ["o", "x", "^", "d", "*"]


@optional_fig_ax
def plot_xy_scatter(
    fig: Figure,
    ax: Axes,
    xy_data: tuple[Iterable[float], Iterable[float]]
    | list[tuple[Iterable[float], Iterable[float]]],
    legend_labels: Iterable[str] | EllipsisType = (),
    legend_title: str | None = None,
    show_line: bool = False,
    cycler: Cycler | Literal["black", "color"] | EllipsisType | None = None,
    **kwargs,
) -> None:
    
    _kwargs = dict(tex=True)
    _kwargs.update(kwargs)

    filter_kwargs(set_axes)(ax, **_kwargs)

    if not isinstance(xy_data, list):
        if cycler is Ellipsis:
            pass
        elif cycler is None:
            __kwargs = dict(color='black', marker="o", s=36.0)
            __kwargs.update(**_kwargs)
            _kwargs = __kwargs
        else:
            ax.set_prop_cycle(cycler)

        if show_line:
            _kwargs.update(linestyle='-')
        else:
            _kwargs.update(linestyle='')

        x_data, y_data = xy_data        
        filter_kwargs(ax.plot, Line2D)(x_data, y_data, **_kwargs)
    else:
        if cycler is None:
            cycler = 'black'
        if isinstance(cycler, str):
            if cycler == 'black':
                cycler = create_cycler(color=["black"] * len(MARKERS), linestyle=["-"], linewidth=[LW] * len(MARKERS))
            elif cycler == "color":
                cycler = create_cycler(color=COLORS, linestyle=["-"] * len(COLORS), linewidth=[LW] * len(COLORS))
            else:
                cmap = getattr(plt.cm, cycler)
                cycler = create_cycler(color=cmap(np.linspace(0, 1, len(xy_data))), linewidth=[LW] * len(xy_data))
                if legend_labels is Ellipsis:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(legend_labels), vmax=max(legend_labels)))
                    divider = make_axes_locatable(ax)
                    ax_cbar = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = fig.colorbar(sm, ax_cbar, shrink=0.5)
                    if legend_title:
                        cbar.set_label(legend_title, rotation=360)

        for i, xy in enumerate(xy_data):
            if i == 0:
                plot_xy_scatter(
                    fig,
                    ax,
                    xy,
                    show_line=show_line,
                    cycler=cycler,
                    **_kwargs,
                )
            else:
                plot_xy_scatter(
                    fig, ax, xy, show_line=show_line, cycler=Ellipsis, **_kwargs,
                )

    if legend_labels:
        filter_kwargs(set_legend)(ax, legend_labels, legend_title, **kwargs)


@optional_fig_ax
def plot_xyz_scatter(
    fig: Figure,
    ax: Axes,
    xyz_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    cmap: str | tuple[str, str] = "viridis",
    s: float | Callable[[np.ndarray], np.ndarray] = 36.0,
    **kwargs,
) -> None:
    
    _kwargs = dict(tex=True)
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
    z_data: np.ndarray, s_min: float = 10.0, s_max: float = 400.0, n: int = 1
) -> np.ndarray:
    """
    Defining a marker size based on the point's z-value

    s(z) = m·zⁿ + c
    """
    m = (s_max - s_min) / (np.max(z_data) ** n - np.min(z_data) ** n)
    c = s_max - m * np.max(z_data) ** n
    return m * z_data + c
