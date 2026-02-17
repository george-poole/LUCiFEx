from typing import ParamSpec, Callable, TypeVar, Any, Iterable, Protocol, Generic

from joblib import Parallel, delayed

from .sim2npy import as_grid_simulation
from .run import run
from .simulation import Simulation


P = ParamSpec('P')
T = TypeVar('T')
def create_and_run(
    factory: Callable[P, Simulation] | Callable[..., Callable[P, Simulation]],
    n_stop: int | None = None,
    t_stop: float | None = None,
    dt_init: float | None = None,
    n_init: int | None = None,
    return_as: Callable[[Simulation], T] | str = 'grid',
) -> Callable[P, T]:
    def _inner(*args: P.args, **kwargs: P.kwargs):
        sim = factory(*args, **kwargs)
        if n_stop is not None or t_stop is not None:
            run(sim, n_stop, t_stop, dt_init, n_init)
        if callable(return_as):
            return return_as(sim)
        elif return_as == 'grid':
            return as_grid_simulation(sim)
        elif return_as == 'tri':
            raise NotImplementedError
        else:
            raise ValueError
    return _inner


T = TypeVar('T')
class ParallelRunCallable(Protocol, Generic[T]):
    def __call__(
        self, 
        **parallel_kws: Iterable[Any],
    ) -> list[T]:
        pass


T = TypeVar('T')
def parallel_run(
    factory: Callable[P, Simulation] | Callable[..., Callable[P, Simulation]],   
    n_proc: int | None,
    n_stop: int | None = None,
    t_stop: float | None = None,
    dt_init: float | None = None,
    n_init: int | None = None,
    return_as: Callable[[Simulation], T] | str = 'grid',
):
    """
    The type `T` returned by `return_as` must be picklable by `joblib`.
    """
    def _(
        *args: P.args, 
        **kwargs: P.kwargs
    ) -> ParallelRunCallable[T]:

        def _inner(
            *a,
            **parallel_kws: Iterable
        ):
            if a:
                raise TypeError('Positional parallel arguments not accepted.')  
            if not parallel_kws:
                raise TypeError('Keyword parallel arguments required.')

            n_kws = [len(v) for v in parallel_kws.values()]
            n_total = n_kws[0]
            if not all(n == n_total for n in n_kws):
                raise TypeError('Keyword parallel arguments must all have the same length.')

            job_func = create_and_run(factory, n_stop, t_stop, dt_init, n_init, return_as)
            if n_proc is None:
                sims = [
                    job_func(*args, **kwargs, **{k: v[n] for k, v in parallel_kws.items()}) for n in range(n_total)
                ]
            else:
                sims = Parallel(n_jobs=n_proc, return_as='list')(
                    delayed(job_func)(*args, **kwargs, **{k: v[n] for k, v in parallel_kws.items()}) for n in range(n_total)
                )
            return sims
        
        return _inner
    
    return _