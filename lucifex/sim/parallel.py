from typing import ParamSpec, Callable, TypeVar, Any, Iterable, Protocol

from joblib import Parallel, delayed

from .sim2npy import as_grid_simulation, GridSimulation, TriSimulation
from .run import run
from .simulation import Simulation


P = ParamSpec('P')
R = TypeVar('R')
def create_and_run(
    factory: Callable[P, R] | Callable[..., Callable[P, R]],
    n_stop: int | None = None,
    t_stop: float | None = None,
    return_as: str | Callable[[Simulation], GridSimulation | TriSimulation | Any] = 'grid',
) -> Callable[P, GridSimulation | TriSimulation | Any]:
    def _inner(*args: P.args, **kwargs: P.kwargs):
        sim = factory(*args, **kwargs)
        run(sim, n_stop, t_stop)
        if callable(return_as):
            return return_as(sim)
        elif return_as == 'grid':
            return as_grid_simulation(sim)
        elif return_as == 'tri':
            raise NotImplementedError
        else:
            raise ValueError
    return _inner


class ParallelRunCallable(Protocol):
    def __call__(
        self, 
        **parallel_kws: Iterable[Any],
    ) -> list[GridSimulation | TriSimulation | Any]:
        pass


def parallel_run(
    factory: Callable[P, R] | Callable[..., Callable[P, R]],   
    n_proc: int | None,
    n_stop: int | None = None,
    t_stop: float | None = None,
    return_as: str | Callable[[Simulation], GridSimulation | TriSimulation | Any] = 'grid',
):
    def _(
        *args: P.args, 
        **kwargs: P.kwargs
    ) -> ParallelRunCallable:

        def _inner(
            *a,
            **parallel_kws: Iterable[Any]
        ) -> list:
            if a:
                raise TypeError('Positional parallel arguments not accepted.')  
            if not parallel_kws:
                raise TypeError('Keyword parallel arguments required.')

            n_kws = [len(v) for v in parallel_kws.values()]
            n_total = n_kws[0]
            if not all(n == n_total for n in n_kws):
                raise TypeError('Keyword parallel arguments must all have the same length.')

            job_func = create_and_run(factory, n_stop, t_stop, return_as)
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