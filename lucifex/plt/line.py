from functools import singledispatch
from typing import Literal, Iterable, Any, Callable
from types import EllipsisType

import numpy as np
from dolfinx.fem import Function
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import Cycler

from ..fem import as_grid_function, GridFunction
from ..utils.py_utils import OverloadTypeError, create_kws_filterer
from .utils import (
    LW, set_legend, create_colorbar, optional_fig_ax, 
    set_axes, create_cycler, optional_multifig_ax,
)


@optional_fig_ax
def plot_line(
    fig: Figure,
    ax: Axes,
    f: Function
    | GridFunction
    | tuple[Iterable[float], Iterable[float]]
    | Iterable[Function | tuple[Iterable[float], Iterable[float]]],
    legend_labels: list[str | float | int] | tuple[float, float] | None = None,
    legend_title: str | None = None,
    cyc: Cycler | Literal["black", "color", "marker", "markerline"] | str | None = None,
    flip: bool = False,
    ax_cbar: Axes | None = None,
    cax: bool = True,
    **kwargs,
) -> None:
    
    if isinstance(f, (Function, GridFunction, tuple)):
        if cyc is not None:
            cyc = create_cycler(cyc)
        _plot_line(f, ax, cyc, flip, **kwargs)
    else:
        if isinstance(legend_labels, tuple):
            if not len(legend_labels) == 2:
                raise TypeError('Legend label must be a two-tuple containing colorbar bounds.')
            mappable = ScalarMappable(
                cmap=cyc, norm=plt.Normalize(vmin=min(legend_labels), vmax=max(legend_labels)),
            )
            cbar = create_colorbar(
                fig, ax, mappable, 
                limits=None, 
                ax_cbar=ax_cbar, 
                cax=cax,
                **dict(shrink=0.5) if cax else {},
            )
            if legend_title:
                fontsize = kwargs.get('legend_fontsizes', 14)
                cbar.set_label(legend_title, rotation=360, ha='left', fontsize=fontsize)

        cyc = create_cycler(cyc, len(f))
        _kwargs = dict(x_lims=None)
        _kwargs.update(kwargs)
        for i, fi in enumerate(f):
            if i == 0:
                _plot_line(fi, ax, cyc=cyc, flip=flip, **_kwargs)
            else:
                _plot_line(fi, ax, cyc=Ellipsis, flip=flip, **_kwargs)

    if legend_labels is not None and not isinstance(legend_labels, tuple):
        create_kws_filterer(set_legend)(ax, legend_labels, legend_title, **kwargs)


@singledispatch
def _plot_line(f, *_, **__):
    raise OverloadTypeError(f)


@_plot_line.register(Function)
def _(
    f: Function,
    ax: Axes,
    cyc: Cycler | None = None,
    flip: bool = False,
    **kwargs,
) -> None:
    f_grid = as_grid_function(f)
    _plot_line(f_grid, ax, cyc, flip, **kwargs)


@_plot_line.register(GridFunction)
def _(
    f: GridFunction,
    ax: Axes,
    cyc: Cycler | None = None,
    flip: bool = False,
    **kwargs,
) -> None:
    xy = (f.mesh.x_axis, f.value)
    return _plot_line(xy, ax, cyc, flip, **kwargs)


@_plot_line.register(tuple)
def _(
    xy: tuple[np.ndarray, np.ndarray],
    ax: Axes,
    cyc: Cycler | EllipsisType | None = None,
    flip: bool = False,
    **kwargs,
) -> None:
    
    x, y = xy
    if flip:
        x, y = y, x

    _kwargs = dict(x_lims=x)
    _kwargs.update(kwargs)
    create_kws_filterer(set_axes)(ax, **_kwargs)
    if cyc is Ellipsis:
        pass
    elif cyc is None:
        __kwargs = dict(linewidth=LW, color='black', linestyle="-")
        __kwargs.update(_kwargs)
        _kwargs = __kwargs
    else:
        ax.set_prop_cycle(cyc)
    create_kws_filterer(ax.plot, Line2D)(x, y, **_kwargs)


@optional_fig_ax
def plot_twin_lines(
    fig: Figure,
    ax: Axes,
    twin_lines: tuple[
        Function | GridFunction | tuple[np.ndarray, np.ndarray], 
        Function | GridFunction | tuple[np.ndarray, np.ndarray]
    ],
    twin_labels: tuple[str | None, str | None] = (None, None),
    twin_kwargs: tuple[dict[str, Any], dict[str, Any]] = None,
    **kwargs,
) -> None:
    _plt_kwargs = {'color': 'black', 'linewidth': LW}
    _plt_kwargs_left = _plt_kwargs | {'linestyle': 'solid'}
    _plt_kwargs_right = _plt_kwargs | {'linestyle': 'dashed'}
    if twin_kwargs is None:
        twin_kwargs = ({}, {})
    _plt_kwargs_left.update(**twin_kwargs[0], **kwargs)
    _plt_kwargs_right.update(**twin_kwargs[1], **kwargs)

    y_label_left, y_label_right = twin_labels
    line_left, line_right = twin_lines
    plot_line(fig, ax, line_left, y_label=y_label_left, **_plt_kwargs_left)
    ax_twin = ax.twinx()
    plot_line(fig, ax_twin, line_right, y_label=y_label_right, **_plt_kwargs_right)


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
    if y_labels is None:
        y_labels = [None] * len(f)

    fig, ax = plt.subplots(len(f), sharex=True, gridspec_kw=gridspec_kw)
    
    for fi, axi, y_label in zip(f, ax, y_labels, strict=True):
        plot_line(fig, axi, fi, y_label=y_label, **plt_kwargs)

    set_axes(ax[0], title=title)
    set_axes(ax[-1], x_label=x_label)
    fig.subplots_adjust(hspace=0)

    return fig, list(ax)


@optional_multifig_ax
def plot_line_multifigure(
    fig: Figure,
    axs_main: list[Axes],
    axs_cbar: list[Axes] | list[tuple[float, float] | None],
    u: Iterable[Function | GridFunction | tuple[Iterable[float], Iterable[float]] | Iterable],
    cyc: Iterable[str] | str | None = None,
    title: Iterable[str] | None = None,
    legend_labels: list[list[str | float | int] | tuple[float, float]] | tuple[float, float] | None = None,
    legend_title: str | None = None,
    *,
    posthook: Callable[[Figure, Axes], None] | None = None,
    **kwargs,
) -> None:

    if isinstance(cyc, str) or cyc is None:
        cyc = [cyc] * len(u)

    if title is None:
        title = [title] * len(u)

    if legend_labels is None or isinstance(legend_labels, tuple):
        legend_labels = [legend_labels] * len(u)

    if isinstance(legend_title, str) or legend_title is None:
        legend_title = [legend_title] * len(u)

    for ui, cy, ttl, leg_lbl, leg_ttl, ax_m, ax_cb in zip(
        u, cyc, title, legend_labels, legend_title, axs_main, axs_cbar
    ):
        if isinstance(ax_cb, tuple):
            _legend_labels = ax_cb
            _legend_title = leg_ttl
            _ax_cbar = ax_m
            _cax = False
        elif isinstance(ax_cb, Axes):
            _legend_labels = leg_lbl
            _legend_title = leg_ttl
            _ax_cbar = ax_cb
            _cax = True
        else:
            if isinstance(leg_lbl, tuple):
                _legend_labels = None
                _legend_title = None
            else: 
                _legend_labels = leg_lbl
                _legend_title = leg_ttl
            _ax_cbar = None
            _cax = True

        plot_line(
            fig, ax_m, ui,
            title=ttl,
            cyc=cy,
            legend_labels=_legend_labels,
            legend_title=_legend_title,
            ax_cbar=_ax_cbar,
            cax=_cax,
            **kwargs,
        )
        if posthook is not None:
            posthook(fig, ax_m)