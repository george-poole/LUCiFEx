import itertools
from typing import ParamSpec, Callable, TypeVar, Iterable, Protocol, Generic

from joblib import Parallel, delayed

from .sim2npy import as_grid_simulation, as_tri_simulation
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
            return as_tri_simulation(sim)
        else:
            raise ValueError(return_as)
    return _inner


T = TypeVar('T')
V = TypeVar('V')
class ParallelRunCallable(Protocol, Generic[T]):
    def __call__(
        self, 
        **parallel_kws: Iterable[V],
    ) -> dict[tuple[V, ...], T] | dict[V, T]:
        ...


T = TypeVar('T')
def parallel_run(
    factory: Callable[P, Simulation] | Callable[..., Callable[P, Simulation]],   
    n_proc: int | None,
    n_stop: int | None = None,
    t_stop: float | None = None,
    dt_init: float | None = None,
    n_init: int | None = None,
    serialize: Callable[[Simulation], T] | str = 'grid',
    link: bool = True,
):
    """
    The type `T` returned by `serialize` must be serializable by `joblib`.
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
             
            if link:
                n_kws = [len(v) for v in parallel_kws.values()]
                n_total = n_kws[0]
                if not all(n == n_total for n in n_kws):
                    raise TypeError('Keyword parallel arguments must all have the same length.')
                job_kws = [{k: v[n] for k, v in parallel_kws.items()} for n in range(n_total)]
            else:
                vals = itertools.product(*([i for i in v] for v in parallel_kws.values()))
                job_kws = [
                    dict(zip(parallel_kws.keys(), v)) for v in vals
                ]

            job_func = create_and_run(factory, n_stop, t_stop, dt_init, n_init, serialize)
            if n_proc is None:
                job_returns = (
                    job_func(*args, **kwargs, **kws) for kws in job_kws
                )
            else:
                job_returns = Parallel(n_jobs=n_proc, return_as='generator')(
                    delayed(job_func)(*args, **kwargs, **kws) for kws in job_kws
                )

            job_keys = [
                i if len(i) > 1 else i[0] for i in (tuple(i.values()) for i in job_kws)
            ]
            return dict(zip(job_keys, job_returns, strict=True))
        
        return _inner
    
    return _