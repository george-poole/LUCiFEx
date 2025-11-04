from inspect import signature
from typing import (
    Callable, Iterable, ParamSpec, Concatenate, 
    Literal, Any, overload,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from cycler import Cycler, cycler


plt.rc("text", usetex=True)
plt.rc("font", family="serif")

LW = 0.75
MS = 4.0
STYLES = ["-", "--", ":", "-.", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]
MARKERS = ["o", "x", "^", "d", "*"]
COLORS = ["black", "blue", "limegreen", "red", "darkorange", "fuchsia"]


P = ParamSpec('P')
def optional_fig(
    plot_func: Callable[Concatenate[Figure, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func)


P = ParamSpec('P')
def optional_ax(
    plot_func: Callable[Concatenate[Axes, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func)


P = ParamSpec('P')
def optional_fig_ax(
    plot_func: Callable[Concatenate[Figure, Axes, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func)


P = ParamSpec('P')
def _optional_fig_and_or_ax(
    plot_func: Callable[Concatenate[Figure, Axes, P], None] | Callable[Concatenate[Figure, P], None] | Callable[Concatenate[Axes, P], None],
):  
    if tuple(signature(plot_func).parameters.values())[0].annotation is Axes:
        ax_only = True
    else:
        ax_only = False

    # @wraps(func)
    def _inner(*args, **kwargs):
        if isinstance(args[0], Figure) and isinstance(args[1], Axes):
            fig, ax, *_args = args
            # routine mutating existing `Figure, Axes` objects and returning `None`
            if ax_only:
                return plot_func(ax, *_args, **kwargs)
            else:
                return plot_func(fig, ax, *_args, **kwargs)
        else:
            # function creating and returning new `Figure, Axes` objects
            fig, ax = plt.subplots()
            if ax_only:
                plot_func(ax, *args, **kwargs)
            else:
                plot_func(fig, ax, *args, **kwargs)
            return fig, ax

    return _inner


@overload
def create_cycler(
    cyc: Cycler | Literal["black", "color", "marker", "markerline"] | str | None,
    num: int | None = None,
) -> Cycler:
    ...


@overload
def create_cycler(
    **kwargs: Any,
) -> Cycler:
    ...


def create_cycler(*args, **kwargs):
    if not args:
        return cycler(**kwargs)
    else:
        return _create_cycler(*args, **kwargs)


def _create_cycler(
    cyc: Cycler | Literal["black", "color", "marker", "markerline"] | str | None,
    num: int | None = None,
) -> Cycler:
    if isinstance(cyc, Cycler):
        return cyc

    if cyc is None:
        cyc = 'black'

    if cyc == 'black':
        ncyc = len(STYLES)
        cyc = cycler(linestyle=STYLES, color=["black"] * ncyc, linewidth=[LW] * ncyc)
    elif cyc == "color":
        ncyc = len(COLORS)
        cyc = cycler(color=COLORS, linestyle=["-"] * ncyc, linewidth=[LW] * ncyc)
    elif cyc in ('marker', 'markerline'):
        if cyc == 'markerline':
            ls = '-'
        else:
            ls = '' 
        ncyc = len(MARKERS)
        cyc = cycler(marker=MARKERS, color=["black"] * ncyc, linewidth=[LW] * ncyc, linestyle=[ls] * ncyc, s=[MS] * ncyc)
    else:
        assert num is not None
        cmap = getattr(plt.cm, cyc)
        cyc = cycler(color=cmap(np.linspace(0, 1, num)), linewidth=[LW] * num)

    return cyc


def set_axes(
    ax: Axes,
    x_lims: tuple[float, float] | Iterable[float] | None = None,
    y_lims: tuple[float, float] | Iterable[float] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    aspect: float | Literal['auto', 'equal'] | None = None,
) -> None:        
    if x_lims is not None:
        ax.set_xlim((np.min(x_lims), np.max(x_lims)))
    if y_lims is not None:
        ax.set_ylim((np.min(y_lims), np.max(y_lims)))
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if aspect is not None:
        ax.set_aspect(aspect)


def set_legend(
    ax: Axes,
    legend_labels: Iterable[str | float | int | None],
    legend_title: str | None = None,
    handles=None,
    loc="upper left",
    bbox_to_anchor=(1, 1),
    frameon=False,
) -> None:    
    legend_labels = [str(i) for i in legend_labels]
    if handles is None:
        args = (legend_labels, )
    else:
        args = (handles, legend_labels)

    ax.legend(
        *args,
        title=legend_title,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon,
    )


def create_mosaic_figure(
    n_rows: int,
    n_cols: int,
    suptitle: str | None = None,
    figscale: float = 1.0,
    indexing: Literal['xy', 'ij', 'ji'] = 'xy',
    **kwargs,
) -> tuple[Figure, np.ndarray]:

    fig: Figure
    fig, axs = plt.subplots(
        n_rows, 
        n_cols,
        figsize=figscale * np.multiply((n_cols, n_rows), plt.rcParams["figure.figsize"]), 
        layout='compressed',
        **kwargs,
    )

    if suptitle:
        ax: Axes = axs.flat[n_cols - 1]
        ax.text(1.0, 1.25, suptitle, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    if indexing == 'xy':
        axs = axs.T[:, ::-1]
    elif indexing == 'ij':
        axs = axs.T

    return fig, axs
