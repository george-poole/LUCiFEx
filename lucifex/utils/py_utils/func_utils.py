from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    overload,
    Any,
)
from collections.abc import (
    Iterable,
    Hashable,
)
from functools import lru_cache, update_wrapper, wraps
from inspect import signature
import time


P = ParamSpec('P')
R = TypeVar('R')
def optional_lru_cache(
    func: Callable[P, R],
):

    @overload
    def _(
        *args: P.args, 
        **kwargs: P.kwargs,
    ) -> R:
        ...

    @overload
    def _(
        *,
        use_cache: bool = False,
        clear_cache: bool = False,
        canonicalize: bool = False,
    ) -> Callable[P, R]:
        ...

    USE = 'use_cache'
    CLEAR = 'clear_cache'
    CANON = 'canonicalize'
    cache_kwargs_default = {USE: False, CLEAR: False, CANON: False}
    
    func_lru_cache = lru_cache(func)
    func_lru_cache_canon = lru_cache(canonicalize_args(func))

    @wraps(func)
    def _(*args, **kwargs):
        if len(args) == 0 and len(kwargs) > 0 and all(i in cache_kwargs_default for i in kwargs):
            _kwargs = cache_kwargs_default.copy()
            _kwargs.update(kwargs)
            if _kwargs[CANON]:
                _func_lru_cache = func_lru_cache_canon
            else:
                _func_lru_cache = func_lru_cache
            if _kwargs[CLEAR]:
                _func_lru_cache.cache_clear()
            if _kwargs[USE]:
                return lambda *a, **k: _func_lru_cache(*a, **k)
            else:
                return lambda *a, **k: func(*a, **k)
        else:
            return _(**cache_kwargs_default)(*args, **kwargs)
    
    return _


P = ParamSpec('P')
R = TypeVar('R')
def log_timing(
    func: Callable[P, R], 
    logged: dict[Hashable, list[float]] | None = None,
    key: Hashable | None = None,
    n: int = 1, 
    overwrite: bool = False,
) -> Callable[P, R]:
    """
    Mutates `logged`
    """
    if key is None:
        key = func.__name__

    n = int(n)
    assert n > 0

    if not overwrite:
        assert key not in logged

    @wraps(func)
    def _(*args: P.args, **kwargs: P.kwargs):
        for _ in range(n):
            t_start = time.perf_counter()
            r = func(*args, **kwargs)
            t_stop = time.perf_counter()
            dt_exec = t_stop - t_start
            if key not in logged:
                logged[key] = []
            logged[key].append(dt_exec)
        return r
    
    return _


P = ParamSpec("P")
R = TypeVar('R')
def replicate_callable(clbl: Callable[P, R]) -> Callable[[Callable], Callable[P, R]]:
    """
    For example, to replicate the callable `clbl` into the function `replica`

    ```
    @replicate_callable(clbl)
    def replica():
        pass
    ```

    or alternatively

    ```
    replica = replicate_callable(clbl)(lambda: None)
    ```
    """
    def _decorator(dummy: Callable[[], None]):
        assert dummy() is None
        def _wrapper(*args, **kwargs):
            return clbl(*args, **kwargs)
        assigned = ["__annotations__"]
        if dummy.__doc__ is None:
            assigned.append("__doc__")
        else:
            _wrapper.__doc__ = dummy.__doc__
        update_wrapper(_wrapper, clbl, assigned)
        _wrapper.__name__ = dummy.__name__
        _wrapper.__qualname__ = dummy.__qualname__
        _wrapper.__defaults__ = clbl.__defaults__ 
        _wrapper.__module__ = clbl.__module__
        return _wrapper
    return _decorator


# TODO get_overloads in Python 3.11+
P = ParamSpec("P")
R = TypeVar('R')
def canonicalize_args(
    func: Callable[P, R],
) -> Callable[P, R]:

    @wraps(func)
    def _(*args, **kwargs):
        sig = signature(func)
        defaults = [v.default for v in sig.parameters.values() if v.default is not v.empty]
        indices = {name: i for i, name in enumerate(sig.parameters)}
                                                            
        _args = []
        _args.extend(args)
        _args.extend(defaults)
        if not len(_args) == len(sig.parameters):
            raise RuntimeError(f'Arguments {_args} cannot be matched to signature {sig.parameters}.')
        for name, value in kwargs.items():
            _args[indices[name]] = value

        return func(*_args)

    return _


P = ParamSpec("P")
R = TypeVar('R')
def filter_kwargs(
    func: Callable[P, R],
    include: Callable | str | Iterable[str | Callable] = (),
    strict: bool = False,
) -> Callable[P, R] | Callable[..., R]:
    def _(*args, **kwargs):
        get_names = lambda f: [
            n for n, p in signature(f).parameters.items() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]
        
        if callable(include) or isinstance(include, str):
            _include = (include, )
        else:
            _include = include

        included_names = []
        for i in _include:
            if callable(i):
                included_names.extend(get_names(i))
            else:
                included_names.append(i)

        if not strict:
            included_names.extend(get_names(func))

        _kwargs = {k: v for k, v in kwargs.items() if k in included_names}

        return func(*args, **_kwargs)

    return _


def arity(
    func: Callable,
    variadic: bool = False,
) -> int:
    params = signature(func).parameters
    if not variadic:
        params = {k: v for k, v in params.items() if v.kind is not (v.VAR_KEYWORD, v.VAR_POSITIONAL)}
    return len(params)


class MultipleDispatchTypeError(TypeError):
    def __init__(
        self,
        arg: Any, 
        sd_func: Callable | None = None
    ):
        msg = f"Unexpected argument type {type(arg)}."
        if sd_func is not None:
            registered_types = tuple(sd_func.registry)[1:]
            msg = f"{msg} Expected one of {registered_types}."
        super().__init__(msg)




