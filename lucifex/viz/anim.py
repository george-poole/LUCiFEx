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
    
