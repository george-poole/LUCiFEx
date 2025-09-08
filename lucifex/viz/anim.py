from typing import Literal, Iterable

import numpy as np
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from dolfinx.fem import Function

from .colormap import plot_colormap, plot_contours


def animate_colormap(
    cmap_series: Iterable[Function | tuple[np.ndarray, np.ndarray, np.ndarray]],
    title_series: str | Iterable[str] | None = None,
    contour_series: Iterable[Function] |None = None,
    # time_slice: StrSlice | None = None,
    axis_names: tuple[str, str] = ("x", "y"),
    colorbar: bool | tuple[float, float] = True,
    aspect: float | Literal["auto", "equal"] = "equal",
    x_lims: tuple[float, float] | None = None,
    y_lims: tuple[float, float] | None = None,
    interval: int = 100,
    **anim_kwargs
) -> FuncAnimation:
    """
    For display in an IPython environment, use
    ```
    from IPython.display import HTML
    HTML(animation.to_html5_video())
    ```
    """
    cmap_series = cmap_series
    n_snapshots = len(cmap_series)

    if title_series is None or isinstance(title_series, str):
        title_series = [title_series] * n_snapshots
    assert len(title_series) == n_snapshots

    if contour_series is not None:
        assert len(contour_series) == n_snapshots

    fig, ax = subplots()

    def _snapshot(n: int) -> tuple[Figure, Axes]:
        plot_colormap(
            fig,
            ax,
            cmap_series[n],
            title_series[n],
            axis_names,
            colorbar,
        )
        if contour_series is not None:
            plot_contours(fig, ax, contour_series[n])
        if x_lims:
            ax.set_xlim(x_lims)
        if y_lims:
            ax.set_ylim(y_lims)
        ax.set_aspect(aspect)

        return fig, ax

    def _initialize() -> tuple[Figure, Axes]:
        return _snapshot(n=0)

    def _update(n: int):
        # removing the figure's previous colormap (and contours)
        [cll.remove() for cll in ax.collections]
        if colorbar is not False:
            # removing the figure's previous colorbar
            fig.axes[1].remove()
        return _snapshot(n)

    return FuncAnimation(fig, _update, n_snapshots, _initialize, interval=interval, **anim_kwargs)
