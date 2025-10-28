from enum import Enum
from collections import defaultdict
from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    overload,
    TypeAlias,
    Any,
    Iterable,
    Hashable,
)
from functools import lru_cache, update_wrapper, wraps
from inspect import signature
import time


class StrEnum(str, Enum):
    def __repr__(self) -> str:
        return repr(self.value)


class classproperty:
    def __init__(self, func):
        self._getfunc = func

    def __get__(self, _, owner):
        return self._getfunc(owner)


P = ParamSpec("P")
R = TypeVar('R')
def replicate_callable(func: Callable[P, R]) -> Callable[[Callable], Callable[P, R]]:
    """
    For example, to replicate the callable `func`

    ```
    @replicate_callable(func)
    def dummy_func():
        pass
    ```
    """
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
        _wrapper.__name__ = dummy.__name__
        _wrapper.__qualname__ = dummy.__qualname__
        _wrapper.__defaults__ = func.__defaults__ 
        _wrapper.__module__ = func.__module__
        return _wrapper
    return _decorator


P = ParamSpec('P')
R = TypeVar('R')
def log_texec(
    func: Callable[P, R], 
    logged: dict[Hashable, list[float]],
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


StrSlice: TypeAlias = str | slice
"""
Type alias for strings representing slices 
e.g. `"start:stop"` or `"start:stop:step"`
"""

COLON = ':'
DOUBLE_COLON = f'{COLON}{COLON}'


# TODO slice[int, int, int] in Python 3.11+
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
    

def nested_dict(
    order: int | None = None,
) -> defaultdict | dict:
    if order is None:
        return defaultdict(nested_dict)
    if order == 1:
        return dict
    if order == 2:
        return defaultdict(dict)
    assert order > 2
    return nested_dict(nested_dict(order - 1))

    
class ToDoError(NotImplementedError):
    def __init__(self):
        super().__init__('Working on it! Coming soon...')