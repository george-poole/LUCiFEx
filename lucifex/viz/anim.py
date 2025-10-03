from typing import (
    Literal, Iterable, Callable, ParamSpec, 
    Concatenate, TypeVar, Protocol, Generic, Any,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from dolfinx.fem import Function


T = TypeVar('T')
class FuncAnimationFromSeries(Protocol, Generic[T]):
    def __call__(
        self, 
        arg_series: list[T],
        **kwargs_series: list[Any],
    ) -> FuncAnimation:
        ...


P = ParamSpec('P')
T = TypeVar('T')
def create_animation(    
    plot_func: Callable[Concatenate[Figure, Axes, T, P], None],
    clear_func: Callable[[Figure, Axes], None] | None = None,
    millisecs: int = 100,
    anim_kwargs: dict | None = None,
    **plt_kwargs,
) -> FuncAnimationFromSeries[T]:
    """
    To display `anim: FunctionAnimation` in an IPython environment
    ```
    from IPython.display import HTML
    HTML(anim.to_html5_video())
    ```
    """
    def _(
        arg_series: list[T],
        **kwargs_series: list[Any],
    ) -> FuncAnimation:
        n_snapshots = len(arg_series)

        fig, ax = plt.subplots()

        def _snapshot(n: int) -> tuple[Figure, Axes]:
            __kwargs_series = {k: v[n] for k, v in kwargs_series.items()}
            plot_func(
                fig,
                ax,
                arg_series[n],
                **__kwargs_series,
                **plt_kwargs
            )
            return fig, ax
        
        _setup = lambda: _snapshot(n=0)

        nonlocal clear_func
        if clear_func is None:
            clear_func = lambda _, ax: [i.remove() for i in (*ax.collections, *ax.lines, *ax.images)]

        def _update(n: int):
            clear_func(fig, ax)
            return _snapshot(n)
        
        nonlocal anim_kwargs
        if anim_kwargs is None:
            anim_kwargs = {}

        return FuncAnimation(
            fig, 
            _update, 
            n_snapshots, 
            _setup, 
            interval=millisecs, 
            **anim_kwargs,
        )
        
    return _
    

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
    from .colormap import plot_colormap, plot_contours
    """
    For display of `animation: FunctionAnimation` in an IPython environment, use
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

    fig, ax = plt.subplots()

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
