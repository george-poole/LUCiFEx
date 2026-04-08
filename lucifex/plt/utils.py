from typing import (
    Callable, Iterable, ParamSpec, Concatenate, 
    Literal, Any, overload,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import Cycler, cycler

from ..utils.py_utils import create_kws_filterer


LW = 0.75
"""
Default linewidth
"""

MS = 4.0
"""
Default marker size
"""

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10))]
MARKERS = ["o", "x", "^", "d", "*"]
COLORS = ["black", "blue", "limegreen", "red", "darkorange", "fuchsia"]


def configure_matplotlib(
    *groups_kws: tuple[str, dict[str, Any]],
    **rcParams: Any,
) -> None:
    """
    Pass `backend='Agg'` for faster plotting without immediate display 
    in an interactive notebook. Saved figures can subsequently be 
    displayed by calling `display_figure`.

    The default backend is `'module://matplotlib_inline.backend_inline'`.
    """
    for grp, kws in groups_kws:
        plt.rc(grp, **kws)
    for k, v in rcParams.items():
        plt.rcParams[k] = v


def reset_matplotlib() -> None:
    import matplotlib
    matplotlib.rcdefaults()


P = ParamSpec('P')
def optional_fig(
    plot_func: Callable[Concatenate[Figure, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None] | Callable[Concatenate[Figure, P], None]:
    return _optional_fig_and_or_ax(plot_func, fig_ax_arg='fig')


P = ParamSpec('P')
def optional_ax(
    plot_func: Callable[Concatenate[Axes, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None] | Callable[Concatenate[Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func, fig_ax_arg='ax')


P = ParamSpec('P')
def optional_fig_ax(
    plot_func: Callable[Concatenate[Figure, Axes, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func, fig_ax_arg='fig_ax')


P = ParamSpec('P')
def _optional_fig_and_or_ax(
    plot_func: Callable[Concatenate[Figure, Axes, P], None] 
    | Callable[Concatenate[Figure, P], None]
    | Callable[Concatenate[Axes, P], None],
    fig_ax_arg: Literal['fig', 'ax', 'fig_ax'],
):  
    # @wraps(func)
    def _inner(*args, **kwargs):
        if isinstance(args[0], Figure) and isinstance(args[1], Axes):
            # mutating existing `Figure, Axes` and returning `None`
            fig, ax, *_args = args
            return_fig_ax = False
        elif isinstance(args[0], Figure) and not isinstance(args[1], Axes):
            # mutating existing `Figure` and returning `None`
            fig, *_args = args
            return_fig_ax = False
        elif isinstance(args[0], Axes):
            # mutating existing `Axes` and returning `None`
            ax, *_args = args
            return_fig_ax = False
        else:
            # creating and returning new `Figure, Axes`
            _args = args
            fig, ax = plt.subplots()
            return_fig_ax = True

        if fig_ax_arg == 'fig':
            plot_func(fig, *_args, **kwargs)
        elif fig_ax_arg == 'ax':
            plot_func(ax, *_args, **kwargs)
        elif fig_ax_arg == 'fig_ax':
            plot_func(fig, ax, *_args, **kwargs)
        else:
            raise ValueError
        
        if return_fig_ax:
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
    **kwargs: Any | list[Any] | tuple[Any, ...],
) -> Cycler:
    ...


def create_cycler(*args, **kwargs):
    if not args:
        iter_types = (list, tuple)
        ncyc = [len(v) for v in kwargs.values() if isinstance(v, iter_types)]
        if not ncyc:
            ncyc = 1
        else:
            ncyc = min(ncyc)
        _kwargs = {k: v if isinstance(v, iter_types) else [v] * ncyc for k, v in kwargs.items()}
        return cycler(**_kwargs)
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
        ncyc = len(LINESTYLES)
        cyc = cycler(linestyle=LINESTYLES, color=["black"] * ncyc, linewidth=[LW] * ncyc)
    elif cyc == "color":
        ncyc = len(COLORS)
        cyc = cycler(color=COLORS, linestyle=["-"] * ncyc, linewidth=[LW] * ncyc)
    elif cyc in ('marker', 'markerline'):
        if cyc == 'markerline':
            ls = '-'
        else:
            ls = '' 
        ncyc = len(MARKERS)
        cyc = cycler(marker=MARKERS, color=["black"] * ncyc, linewidth=[LW] * ncyc, linestyle=[ls] * ncyc, ms=[MS] * ncyc)
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
    fontsizes: float | tuple[float, float, float] = (16.0, 14.0, 14.0),
    aspect: float | Literal['auto', 'equal'] | None = None,
) -> None:     

    if isinstance(fontsizes, (float, int)):
        fontsizes = tuple([fontsizes] * 3)
    title_fsz, x_label_fsz, y_label_fsz = fontsizes


    if x_lims is not None:
        ax.set_xlim((np.min(x_lims), np.max(x_lims)))
    if y_lims is not None:
        ax.set_ylim((np.min(y_lims), np.max(y_lims)))
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=x_label_fsz)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=y_label_fsz)
    if title is not None:
        ax.set_title(title, fontsize=title_fsz)
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
    legend_fontsizes: float | tuple[float, float] = (14.0, 12.0),
    legend_title_alignment: str | None = None,
) -> None:    
    legend_labels = [str(i) for i in legend_labels]
    if handles is None:
        args = (legend_labels, )
    else:
        args = (handles, legend_labels)

    if isinstance(legend_fontsizes, (float, int)):
        legend_fontsizes = tuple([legend_fontsizes] * 3)

    title_fontsize, label_fontsize = legend_fontsizes

    lgnd = ax.legend(
        *args,
        title=legend_title,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon,
        title_fontsize=title_fontsize,
        fontsize=label_fontsize,
    )
    if legend_title_alignment is None:
        lgnd.get_title().set_multialignment("center")


def create_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable,
    limits: tuple[float, float] | None,
    cbar_ax: Axes | None,
    cax: bool = True,
    cbar_title: str | None = None,
    cbar_title_kws: dict | None = None,
    **kwargs,
) -> Colorbar:
    if cbar_ax is None:
        _kwargs = dict(
            position="right", size="5%", pad=0.1,
        )
        _kwargs.update(**kwargs)
        divider = make_axes_locatable(ax)
        cbar_ax = create_kws_filterer(divider.append_axes)(**_kwargs)
    if cax:
        cbar = fig.colorbar(mappable, cax=cbar_ax, **kwargs)
    else:
        cbar = fig.colorbar(mappable, ax=cbar_ax, **kwargs)

    if limits is not None:
        assert len(limits) == 2
        mappable.set_clim(*limits)

    if cbar_title:
        cbar_title_kws = {} if cbar_title_kws is None else cbar_title_kws
        _kws = dict(
            rotation=360, ha='left', fontsize=14,
        )
        _kws.update(**cbar_title_kws) 
        cbar.set_label(cbar_title, **_kws)
 
    return cbar


def create_multifigure(
    n_rows: int = 1,
    n_cols: int = 1,
    cbars: bool | tuple[float, float] = False,
    figscale: float = 1.0,
    width_ratio: float = 0.025,
    suptitle: str | None = None,
    suptitle_kws: dict | None = None,
    suptitle_index: int | None = None,
    **subplots_kwargs,
) -> tuple[Figure, list[Axes], list[Axes] | list[tuple[float, float] | None]]:

    _subplots_kwargs = dict(
        figsize=figscale * np.multiply((n_cols, n_rows), plt.rcParams["figure.figsize"]), 
        layout='compressed',
    )
    if cbars is True:
        _subplots_kwargs.update({'width_ratios': np.array([(1, width_ratio)] * n_cols).flatten()})
        n_cols = 2 * n_cols

    _subplots_kwargs.update(subplots_kwargs)

    fig, _ = plt.subplots(n_rows, n_cols, **_subplots_kwargs)

    if suptitle:
        if suptitle_kws is None:
            suptitle_kws = {}
        if suptitle_index is None:
            if cbars is True:
                suptitle_index = n_cols - 2
            else:
                suptitle_index = n_cols - 1
        axs_sup: Axes = fig.axes[suptitle_index]
        _suptitle_kws = dict(
            x=1.0,
            y=1.25,
            horizontalalignment='right', 
            verticalalignment='bottom', 
            transform=axs_sup.transAxes,
            fontsize=16,
        )
        _suptitle_kws.update(suptitle_kws)
        axs_sup.text(s=suptitle, **_suptitle_kws)

    if cbars is True:
        axs_main = fig.axes[0::2]
        axs_cbar = fig.axes[1::2]
    else:
        axs_main = fig.axes
        axs_cbar = [None] * len(axs_main)
        if isinstance(cbars, tuple):
            axs_cbar[n_cols - 1] = cbars

    return fig, axs_main, axs_cbar


P = ParamSpec('P')
def optional_multifig_ax(
    plot_func: Callable[Concatenate[Figure, list[Axes], list[Axes | tuple[float, float] | None], P], None],
):

    @overload
    def _(
        **create_multifigure_kws: dict,
    ) -> Callable[P, tuple[Figure, list[Axes], list[Axes | tuple[float, float] | None]]]:
        ...

    @overload
    def _(
        **create_multifigure_kws: dict,
    ) -> Callable[Concatenate[Figure, list[Axes], list[Axes | tuple[float, float] | None], P], None]:
        ...
    
    @overload
    def _(
        fig: Figure,
        axs_main: list[Axes],
        axs_cbar: list[Axes] | list[tuple[float, float] | None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        ...

    N_COLS = 'n_cols'
    N_ROWS = 'n_rows'

    def _(*args, **kwargs):
        if not args and kwargs:
            def _inner(*a, **k):
                if isinstance(a[0], Figure):
                    return plot_func(*a, **k)
                else:
                    plt_series = a[0]
                    n_cols = kwargs.get(N_COLS)
                    n_rows = kwargs.get(N_ROWS)
                    assert (n_cols, n_rows).count(None) != 2
                    if n_cols is None:
                        n_cols = len(plt_series) // n_rows
                    if n_rows is None:
                        n_rows = len(plt_series) // n_cols
                    fig, axs_main, axs_cbar = create_multifigure(
                        n_rows, 
                        n_cols, 
                        **{k: v for k, v in kwargs.items() if not k in (N_COLS, N_ROWS)}
                    )
                    plot_func(fig, axs_main, axs_cbar, *a, **k)
                    return fig, axs_main, axs_cbar
            return _inner
        elif isinstance(args[0], Figure):
            fig, axs_main, axs_cbar, *_args = args
            return plot_func(fig, axs_main, axs_cbar, *_args, **kwargs)
        else:
            raise TypeError

    return _
