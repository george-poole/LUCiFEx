import os
from typing import (overload, Callable, ParamSpec, Any, TypeVar, Protocol, Generic, TypeAlias, Any)
from functools import wraps
from collections.abc import Iterable

from .write import write
from .post_proxy import Proxy, CoProxy


@overload
def load_from_proxy(
    directory: str,
    args: tuple[Proxy | Any, ...],
    kwargs: dict[str, Proxy | Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    ...

@overload   
def load_from_proxy(
    directories: Iterable[str],
    args: tuple[CoProxy | Any, ...],
    kwargs: dict[str, CoProxy | Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    ...
    

def load_from_proxy(
    directory,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):

    _args = []
    _kwargs = {}

    for arg in args:
        if isinstance(arg, (Proxy, CoProxy)):
            arg = arg.load_arg(directory)
        _args.append(arg)

    for name, value in kwargs.items():
        if isinstance(value, (Proxy, CoProxy)):
            value = value.load_arg(directory)
        _kwargs[name] = value
    
    return tuple(_args), _kwargs    

WriteArg: TypeAlias = str | Any

R = TypeVar('R')
class ProxyCallable(Generic[R], Protocol):
    def __call__(
        self, 
        *args: Proxy | Any, 
        **kwargs: Proxy | Any,
    ) -> R:
        ...


P = ParamSpec('P')
R = TypeVar('R')
def postprocess(
    func: Callable[P, R],
):
    """
    Decorator for postprocess functions operating on data from one simulation directory only to compute 
    some combined quantity or figure.

    
    e.g.\\
    `fig, ax = plot_func(u, ...)`

    `fig, ax = plot_func(dir_path)((u_name, u_file, ...), ...)`
    returns the postprocessed object.
    
    `plot_func(dir_path, plot_file, plot_dir_path)((u_name, u_file, *u_load_args), ...)`
    returns `None` and saves the postprocessed object.

    """

    # TODO assert func has full type annotations
    # TODO assert func 0th is not str or Iterable[str]

    @overload
    def _(*args: P.args, **kwargs: P.kwargs) -> R:
        ...

    @overload
    def _(
        directory: str,
        /,
    ) -> Callable[P, R] | ProxyCallable[R]:
        ...

    @overload
    def _(
        directory: str,
        /,
        *write_args: WriteArg,
        **write_kwargs: WriteArg,
    ) -> Callable[P, None] | ProxyCallable[None]:
        ...

    @overload
    def _(
        directories: Iterable[str],
        *write_args: WriteArg,
        **write_kwargs: WriteArg,
    ) -> Callable[P, None] | ProxyCallable[None]:
        ...

    @wraps(func)    
    def _(*args, **kwargs):
        if isinstance(args[0], str):
            directory, write_args = args[0], args[1:]
            write_kwargs = kwargs

            def _inner(*a, **k):
                _a, _k = load_from_proxy(directory, a, k)
                rtrn = func(*_a, **_k)

                nonlocal write_args
                nonlocal write_kwargs
                if write_args or write_kwargs:
                    DIR_PATH = 'dir_path'
                    join_dir_path = lambda s: os.path.join(directory, s) if isinstance(s, str) else s
                    if len(write_args) >= 2:
                        dir_path = join_dir_path(write_args[1])
                        write_args = (write_args[0], dir_path, *write_args[2:])
                    elif DIR_PATH in write_kwargs:
                        dir_path = join_dir_path(write_kwargs[DIR_PATH])
                        write_kwargs[DIR_PATH] = dir_path
                    else:
                        write_kwargs[DIR_PATH] = directory
                    write(rtrn, *write_args, **write_kwargs)
                    print('Postprocess result saved.')
                else:
                    return rtrn
            
            return _inner
        
        elif isinstance(args[0], Iterable) and all(isinstance(i, str) for i in args[0]):
            directories, *write_args = args
            write_kwargs = kwargs

            FILE_NAME = 'file_name'
            if len(write_args) >= 1:
                file_name, *write_args = write_args
            elif FILE_NAME in write_kwargs:
                file_name = write_kwargs[FILE_NAME]
                write_kwargs = {k: v for k, v in write_kwargs.items() if k != FILE_NAME}
            else:
                file_name = None

            def _inner(*a, **k):
                [_(d, file_name, *write_args, **write_kwargs)(*a, **k) for d in directories]

            return _inner
    
        else:
            return func(*args, **kwargs)

    return _


R = TypeVar('R')
class CoProxyCallable(Generic[R], Protocol):
    def __call__(
        self, 
        *args: CoProxy | Any, 
        **kwargs: CoProxy | Any,
    ) -> R:
        ...


P = ParamSpec('P')
R = TypeVar('R')
def co_postprocess(
    func: Callable[P, R],
): 
    """
    Decorator for postprocess functions operating on data from multiple simulation directories
    to compute some combined quantity or figure.

    e.g.\\
    `fig, ax = plot_func([u_a, u_b, u_c, ...], ...)`

    `fig, ax = plot_func([dir_path_a, dir_path_b, dir_path_c, ...])((u_name, u_file, *u_load_args), ...)` 
    returns the postprocessed object

    `plot_func([dir_path_a, dir_path_b, dir_path_c], plot_file, plot_dir_path)((u_name, u_file, ...), ...)`
    returns `None` and saves the postprocessed object.
    """

    @overload
    def _(*args: P.args, **kwargs: P.kwargs) -> R:
        ...

    @overload
    def _(
        directories: Iterable[str],
        /,
        *write_args: WriteArg,
        **write_kwargs: WriteArg,
    ) -> Callable[P, None] | CoProxyCallable[None]:
        ...

    @wraps(func)    
    def _(*args, **kwargs):
        if isinstance(args[0], Iterable) and all(isinstance(i, str) for i in args[0]):
            directories, write_args = args[0], args[1:]
            write_kwargs = kwargs

            def _inner(*a, **k):
                _a, _k = load_from_proxy(directories, a, k)
                rtrn = func(*_a, **_k)  

                if write_args or write_kwargs:
                    write(rtrn, *write_args, **write_kwargs)
                    print('Postprocess result saved.')
                else:
                    return rtrn

            return _inner

        else:
            return func(*args, **kwargs)
        
    return _