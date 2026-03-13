import itertools
from typing import ParamSpec, Callable, TypeVar, Iterable, Protocol, Generic

from joblib import Parallel, delayed

from .sim2npy import as_npy_simulation
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
    serialize: Callable[[Simulation], T] = as_npy_simulation,
) -> Callable[P, T]:
    def _inner(*args: P.args, **kwargs: P.kwargs):
        sim = factory(*args, **kwargs)
        if n_stop is not None or t_stop is not None:
            run(sim, n_stop, t_stop, dt_init, n_init)
        return serialize(sim)
    return _inner


T = TypeVar('T')
V = TypeVar('V')
class ParallelRunCallable(Protocol, Generic[T]):
    def __call__(
        self, 
        **kws_opts: Iterable[V],
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
    serialize: Callable[[Simulation], T] = as_npy_simulation,
    link: bool = True,
    **joblib_kws,
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
            **kws_opts: Iterable
        ):
            if a:
                raise TypeError('Positional parallel arguments not accepted.')  
            if not kws_opts:
                raise TypeError('Keyword parallel arguments required.')
             
            if link:
                n_kws = [len(v) for v in kws_opts.values()]
                n_total = n_kws[0]
                if not all(n == n_total for n in n_kws):
                    raise TypeError('Keyword parallel arguments must all have the same length.')
                job_kws = [{k: v[n] for k, v in kws_opts.items()} for n in range(n_total)]
            else:
                job_kws = [
                    dict(zip(kws_opts.keys(), v)) for v in itertools.product(*kws_opts.values())
                ]

            job_func = create_and_run(factory, n_stop, t_stop, dt_init, n_init, serialize)
            if n_proc is None:
                job_returns = (
                    job_func(*args, **kwargs, **kws) for kws in job_kws
                )
            else:
                job_returns = Parallel(n_jobs=n_proc, return_as='generator', **joblib_kws)(
                    delayed(job_func)(*args, **kwargs, **kws) for kws in job_kws
                )

            job_keys = [
                i if len(i) > 1 else i[0] for i in (tuple(i.values()) for i in job_kws)
            ]
            return dict(zip(job_keys, job_returns, strict=True))
        
        return _inner
    
    return _


T = TypeVar('T')
def combine_options(
    *opts: Iterable[T],
    link: bool,
    return_as: type = list,
) -> Iterable[tuple[T, ...]]:
    if link:
        gen = zip(*opts)
    else:
        gen = itertools.product(*opts)
    return return_as(gen)

