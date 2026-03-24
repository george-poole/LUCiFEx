from typing import (
    Callable, ParamSpec, Concatenate, 
    TypeVar, Protocol, Generic, Any,
)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation


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
    plot: Callable[Concatenate[Figure, Axes, T, P], None]
    | Callable[Concatenate[Figure, list, list, T, P], None],
    clear_func: Callable[[Figure], None] | None = None,
    millisecs: int = 100,
    anim_kwargs: dict | None = None,
    close: bool = True,
    **plot_kwargs,
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
        assert all(len(i) == n_snapshots for i in kwargs_series.values())

        def _snapshot(
            n: int,
            setup: bool = False,
        ) -> tuple[Figure, ...]:
            _kwargs_series = {k: v[n] for k, v in kwargs_series.items()}
            if setup:
                return plot(
                    arg_series[n],
                    **_kwargs_series,
                    **plot_kwargs
                )
            else:
                plot(
                    fig, 
                    *ax_objects,
                    arg_series[n],
                    **_kwargs_series,
                    **plot_kwargs
                )
                return fig, *ax_objects
            
        fig, *ax_objects = _snapshot(0, setup=True)
        
        _setup = lambda: _snapshot(n=0)

        nonlocal clear_func
        if clear_func is None:

            CBAR_ATTR = '_colorbar'
            def _clear_ax(ax: Axes) -> None:
                if hasattr(ax, CBAR_ATTR):
                    cbar = getattr(ax, CBAR_ATTR)
                    cbar.ax.clear()
                    cbar.ax.set_axis_off()
                objs = (*ax.collections, *ax.lines, *ax.images)
                [i.remove() for i in objs]
                    
            clear_func = lambda fig: [
                _clear_ax(ax) for ax in fig.axes 
            ]

        def _update(n: int):
            clear_func(fig)
            return _snapshot(n)
        
        nonlocal anim_kwargs
        if anim_kwargs is None:
            anim_kwargs = {}

        anim = FuncAnimation(
            fig, 
            _update, 
            n_snapshots, 
            _setup, 
            interval=millisecs, 
            **anim_kwargs,
        )
        if close:
            plt.close(fig)

        return anim
        
    return _
    
