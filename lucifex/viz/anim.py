from typing import Literal, Iterable

import numpy as np
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from dolfinx.fem import Function

from .colormap import plot_colormap, plot_contours


def animate_line(
    lines: Iterable[Function | tuple[np.ndarray, np.ndarray]],
    titles: str | Iterable[str] | None = None,
):
    ...


def animate_colormap(
    cmaps: Iterable[Function | tuple[np.ndarray, np.ndarray, np.ndarray]],
    titles: str | Iterable[str] | None = None,
    contours: Iterable[Function] |None = None,
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
    cmaps = cmaps
    n_snapshots = len(cmaps)

    if titles is None or isinstance(titles, str):
        titles = [titles] * n_snapshots
    assert len(titles) == n_snapshots

    if contours is not None:
        assert len(contours) == n_snapshots

    fig, ax = subplots()

    def _snapshot(n: int) -> tuple[Figure, Axes]:
        plot_colormap(
            fig,
            ax,
            cmaps[n],
            titles[n],
            axis_names,
            colorbar,
        )
        if contours is not None:
            plot_contours(fig, ax, contours[n])
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
