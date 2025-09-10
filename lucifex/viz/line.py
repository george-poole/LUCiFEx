from functools import singledispatch
from typing import Literal, Callable, Iterable, Any
from types import EllipsisType

import numpy as np
from dolfinx.fem import Function
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import Cycler, cycler as create_cycler

from ..utils import grid, MultipleDispatchTypeError, filter_kwargs
from .utils import COLORS, LW, set_legend, optional_ax, optional_fig_ax, set_axes, texify


# solid, dashed, dotted, dashdotted, dashdotdotted, loosely dashdotdotted
STYLES = ["-", "--", ":", "-.", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]

@optional_fig_ax
def plot_line(
    fig: Figure,
    ax: Axes,
    f: Function
    | tuple[Iterable[float], Iterable[float]]
    | Iterable[Function | tuple[Iterable[float], Iterable[float]]],
    legend_labels: list[str | float | int] | tuple[float, float] | None = None,
    legend_title: str | None = None,
    cycler: Cycler | Literal["black", "color"] | str | None = None,
    **kwargs,
) -> None:
    if isinstance(f, (Function, tuple)):
        _plot_line(f, ax, cycler, **kwargs)
    else:
        if cycler is None:
            cycler = 'black'
        if isinstance(cycler, str):
            if cycler == 'black':
                cycler = create_cycler(color=["black"] * len(STYLES), linestyle=STYLES, linewidth=[LW] * len(STYLES))
            elif cycler == "color":
                cycler = create_cycler(color=COLORS, linestyle=["-"] * len(COLORS), linewidth=[LW] * len(COLORS))
            else:
                if isinstance(f, (Function, tuple)):
                    raise TypeError('Colormap cycler requires argument type `f: Iterable[Function | tuple[Iterable[float], Iterable[float]]]`')
                cmap = getattr(plt.cm, cycler)
                cycler = create_cycler(color=cmap(np.linspace(0, 1, len(f))), linewidth=[LW] * len(f))
                if isinstance(legend_labels, tuple):
                    assert len(legend_labels) == 2
                    mappable = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(legend_labels), vmax=max(legend_labels)))
                    divider = make_axes_locatable(ax)
                    ax_cbar = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = fig.colorbar(mappable, ax_cbar, shrink=0.5)
                    if legend_title:
                        if kwargs.get('tex') is not False:
                            legend_title = texify(legend_title)
                        cbar.set_label(legend_title, rotation=360, ha='left')
        
        _kwargs = dict(x_lims=None)
        _kwargs.update(kwargs)
        for i, fi in enumerate(f):
            if i == 0:
                _plot_line(fi, ax, cycler=cycler, **_kwargs)
            else:
                _plot_line(fi, ax, cycler=Ellipsis, **_kwargs)

        if legend_labels is not None and not isinstance(legend_labels, tuple):
            filter_kwargs(set_legend)(ax, legend_labels, legend_title, **_kwargs)


@singledispatch
def _plot_line(f):
    raise MultipleDispatchTypeError(f)


@_plot_line.register(Function)
def _(
    f: Function,
    ax: Axes,
    cycler: Cycler | None = None,
    **kwargs,
) -> None:
    (x, ) = grid(f.function_space.mesh)
    f_grid  = grid(f)
    _plot_line((x, f_grid), ax, cycler, **kwargs)


@_plot_line.register(tuple)
def _(
    xy: tuple[np.ndarray, np.ndarray],
    ax: Axes,
    cycler: Cycler | EllipsisType | None = None,
    **kwargs,
) -> None:
    
    x, y = xy

    _kwargs = dict(x_lims=x, tex=True)
    _kwargs.update(**kwargs)

    filter_kwargs(set_axes)(ax, **_kwargs)
    if cycler is Ellipsis:
        pass
    elif cycler is None:
        __kwargs = dict(linewidth=LW, color='black', linestyle="-")
        __kwargs.update(**_kwargs)
        _kwargs = __kwargs
    else:
        ax.set_prop_cycle(cycler)
    filter_kwargs(ax.plot, Line2D)(x, y, **_kwargs)


@optional_ax
def plot_twin_lines(
    ax: Axes,
    x: np.ndarray | tuple[np.ndarray, np.ndarray],
    y: tuple[np.ndarray, np.ndarray],
    y_labels: tuple[str | None, str | None] = (None, None),
    twin_kwargs: tuple[dict[str, Any], dict[str, Any]] = None,
    **kwargs,
) -> None:
    _plt_kwargs = {'color': 'black', 'linewidth': LW}

    _plt_kwargs_left = _plt_kwargs | {'linestyle': 'solid'}
    _plt_kwargs_right = _plt_kwargs | {'linestyle': 'dashed'}
    if twin_kwargs is None:
        twin_kwargs = ({}, {})
    _plt_kwargs_left.update(twin_kwargs[0])
    _plt_kwargs_right.update(twin_kwargs[1])

    y_left, y_right = y
    if not isinstance(x, tuple):
        x = (x, x)
    x_left, x_right = x
    ax.plot(x_left, y_left, **_plt_kwargs_left)

    ax_twin = ax.twinx()
    ax_twin.plot(x_right, y_right, **_plt_kwargs_right)

    y_label_left, y_label_right = y_labels
    _kwargs = dict(tex=True)
    _kwargs.update(kwargs)
    filter_kwargs(set_axes)(ax, x, y_label=y_label_left, **_kwargs)
    filter_kwargs(set_axes)(ax_twin, y_label=y_label_right, **_kwargs)


def plot_stacked_lines(
    f: Iterable[Function | tuple[Iterable[float], Iterable[float]]],
    x_label: str | None = None,
    y_labels: Iterable[str] | None = None,
    title: str | None = None,
    gridspec_kw: dict | None = None,
    **plt_kwargs,
) -> tuple[Figure, list[Axes]]:
    assert len(f) > 0
    if gridspec_kw is None:
        gridspec_kw = {}

    fig, ax = plt.subplots(len(f), sharex=True, gridspec_kw=gridspec_kw)
    
    for fi, axi, y_label in zip(f, ax, y_labels, strict=True):
        plot_line(fig, axi, fi, y_label=y_label, **plt_kwargs)

    set_axes(ax[0], title=title)
    if plt_kwargs.get('tex') is not False:
        x_label = texify(x_label)
    set_axes(ax[-1], x_label=x_label)
    fig.subplots_adjust(hspace=0)

    return fig, list(ax)