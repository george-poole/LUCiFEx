from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    overload,
    TypeAlias,
    Any,
    Iterable,
)
from functools import lru_cache, update_wrapper, wraps
from inspect import signature
import time


class classproperty:
    def __init__(self, func):
        self._getfunc = func

    def __get__(self, _, owner):
        return self._getfunc(owner)


P = ParamSpec("P")
R = TypeVar('R')
def copy_callable(func: Callable[P, R]) -> Callable[[Callable], Callable[P, R]]:
    def _decorator(dummy: Callable[[], None]):
        assert dummy() is None
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        assigned = ["__annotations__"]
        if dummy.__doc__ is None:
            assigned.append("__doc__")
        else:
            _wrapper.__doc__ = dummy.__doc__
        update_wrapper(_wrapper, func, assigned)
        _wrapper.__name__ = dummy.__name__ # TODO __qualname__ too?
        _wrapper.__defaults__ = func.__defaults__ 
        _wrapper.__module__ = func.__module__
        return _wrapper
    return _decorator


P = ParamSpec('P')
R = TypeVar('R')
def log_texec(
    func: Callable[P, R], 
    logged: dict[str, list[float]],
    name: str | None = None, 
) -> Callable[P, R]:
    """
    Mutates `logged`
    """
    
    if name is None:
        name = func.__name__

    assert name not in logged

    @wraps(func)
    def _(*args: P.args, **kwargs: P.kwargs):
        t_start = time.perf_counter()
        r = func(*args, **kwargs)
        t_stop = time.perf_counter()
        dt_exec = t_stop - t_start
        if name not in logged:
            logged[name] = []
        logged[name].append(dt_exec)
        return r
    
    return _

    
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
    ) -> Callable[P, R]:
        ...

    func_lru_cache = lru_cache(func)
    USE = 'use_cache'
    CLEAR = 'clear_cache'
    cache_kwargs_default = {USE: False, CLEAR: False}
        
    @wraps(func)
    def _(*args, **kwargs):
        if len(args) == 0 and len(kwargs) > 0 and all(i in cache_kwargs_default for i in kwargs):
            cache_kwargs = cache_kwargs_default.copy()
            cache_kwargs.update(kwargs)
            if cache_kwargs[CLEAR]:
                func_lru_cache.cache_clear()
            if cache_kwargs[USE]:
                return lambda *a, **k: func_lru_cache(*a, **k)
            else:
                return lambda *a, **k: func(*a, **k)
        else:
            return _(**cache_kwargs_default)(*args, **kwargs)
    
    return _


P = ParamSpec("P")
R = TypeVar('R')
def filter_kwargs(
    func: Callable[P, R],
    include: Callable | Iterable[str] = (),
    strict: bool = False,
) -> Callable[P, R] | Callable[..., R]:
    def _(*args, **kwargs):
        get_names = lambda f: [n for n, p in signature(f).parameters.items() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
        
        if callable(include):
            included_names = get_names(include)
        else:
            included_names = list(include)

        if not strict:
            included_names.extend(get_names(func))

        _kwargs = {k: v for k, v in kwargs.items() if k in included_names}
        return func(*args, **_kwargs)

    return _

    #  is_variadic = lambda f: Parameter.VAR_KEYWORD in [p.kind for p in signature(func).parameters.values()]    
    #####
    #     excluded_names = [n for f in exclude for n in get_names(f)]
    #     kwargs = {k: v for k, v in kwargs.items() if k not in excluded_names}
    #     if not strict:
    #         for p in signature(func).parameters.values():
    #             if p.kind is p.VAR_KEYWORD:
    #                 return func(*args, **kwargs)
    #     kwargs = {k: v for k, v in kwargs.items() if k in get_names(func)}
    #     return func(*args, **kwargs)
    # return _



def MultipleDispatchTypeError(arg, sd_func: Callable | None = None) -> TypeError:
    msg = f"Unexpected type {type(arg)}."
    if sd_func is not None:
        registered_types = tuple(sd_func.registry)[1:]
        msg = f"{msg} Expected one of {registered_types}."
    return TypeError(msg)


StrSlice: TypeAlias = str | slice
"""
Type alias for strings representing slices 
e.g. `"start:stop"` or `"start:stop:step"`
"""

COLON = ':'
DOUBLE_COLON = f'{COLON}{COLON}'

# TODO slice[int, int, int] Python 3.11+
def as_slice(s: str | slice) -> slice:
    if not is_slice(s):
        raise ValueError(f'Invalid string {s} representing a slice.')

    if isinstance(s, slice):
        return s
    if isinstance(s, str):
        s = s.replace(' ', '')
        n_colon = s.count(COLON)
        if n_colon == 1:
            start, stop = s.split(COLON)
            step = ''
        elif n_colon == 2:
            if DOUBLE_COLON in s:
                start, step = s.split(DOUBLE_COLON)
                stop = ''
            else:
                start, stop, step = s.split(COLON)
        else:
            raise ValueError(f'Expected 1 or 2 colons in the string representing a slice.')
        
        if start == '':
            start = 0
        else:
            start = int(start)

        if stop == '':
            stop = None
        else:
            stop = int(stop)

        if step == '':
            step = None
        else:
            step = int(step)

        return slice(start, stop, step)
    
    raise MultipleDispatchTypeError(s)


def is_slice(s: str | slice | Any) -> bool:
    if isinstance(s, slice):
        return True
    elif isinstance(s, str):
        return s.count(COLON) >= 1 and s.count(COLON) <= 2
    else:
        return False
    

# Start: TypeAlias = int
# Stop: TypeAlias = int
# Step: TypeAlias = int
# def as_range(
#     arg: tuple[Step, Stop] | tuple[Start, Step, Stop] | range | list[int],
# ) -> list[int] | range:
    
#     if isinstance(arg, tuple):
#         match arg:
#             case step, stop:
#                 step, stop = arg
#                 start = 0

#             case start, step, stop:
#                 step, step, stop = arg

#         return range(start, stop, step)
    
#     else:
#         if not isinstance(arg, (range, list)):
#             raise TypeError
#         return arg