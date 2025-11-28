from typing import overload, Callable, Any, TypeAlias, Any
from typing_extensions import Unpack
from collections.abc import Iterable
from functools import singledispatch

from .load import load


ObjectName: TypeAlias = str
ObjectType: TypeAlias = type
FileName: TypeAlias = str

class Proxy:
    id_func = staticmethod(lambda *a: a[0] if len(a) == 1 else a)
    """
    `id(x) -> x` \\
    `id(x, y, ...) -> (x, y, ...)`
    `id([x, y, ...]) -> [x, y, ...]`
    """

    def __init__(
        self, 
        func: Callable[[Unpack[tuple]], Any],
        objs: list[tuple[ObjectName, ObjectType, FileName, Unpack[tuple]]], 
        use_cache: bool = True,
        clear_cache: bool = False
    ):
        self._func = func
        self._objs = objs
        self._use_cache = use_cache
        self._clear_cache = clear_cache

    def load_arg(
        self, 
        directory: str, 
        func : Callable[[Unpack[tuple]], Any] | None = None,
    ) -> Any:
        
        objs_loaded = []
        for (name, tp, file_name, *args) in self._objs:
            obj_loaded = load(tp, self._use_cache, self._clear_cache)(name, directory, file_name, *args)
            objs_loaded.append(obj_loaded)

        if func is None:
            func = self._func

        return func(*objs_loaded)


@overload
def proxy(
    obj: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]], 
    *,
    use_cache: bool = True,
    clear_cache: bool = False
) -> Proxy:
    """
    `proxy('x') -> x`
    """
    ...


@overload
def proxy(
    func: Callable[[Any], Any], 
    obj: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]], 
    *,
    use_cache: bool = True,
    clear_cache: bool = False
) -> Proxy:
    """
    `proxy(f, 'x') -> f(x)`
    """
    ...


@overload
def proxy(
    obj: list[tuple[ObjectName, ObjectType, FileName, Unpack[tuple]]], 
    *,
    use_cache: bool = True,
    clear_cache: bool = False
) -> Proxy:
    """
    `proxy(['x', 'y', ...]) -> (x, y, ...)`
    """
    ...


@overload
def proxy(
    func: Callable[..., Any], 
    obj: list[tuple[ObjectName, ObjectType, FileName, Unpack[tuple]]],
    *, 
    use_cache: bool = True,
    clear_cache: bool = False
) -> Proxy:
    """
    `proxy(f, ['x', 'y', ...]) -> f(x, y, ...)`
    """
    ...


@singledispatch
def proxy(
    func, 
    objs, 
    use_cache: bool = True,
    clear_cache: bool = False,
):
    assert callable(func)
    if not isinstance(objs, list):
        objs = [objs]
    return Proxy(func, objs, use_cache, clear_cache)


@proxy.register(tuple)
def _(obj, **kwargs):
    return proxy(Proxy.id_func, obj, **kwargs)


@proxy.register(list)
def _(obj, **kwargs):
    return proxy(Proxy.id_func, obj, **kwargs)


class CoProxy(Proxy):
    def __init__(
        self, 
        func: Callable[[Unpack[tuple]], Any],
        objs: list[tuple[ObjectName, ObjectType, FileName, Unpack[tuple]]], 
        use_cache: bool = True,
        clear_cache: bool = False,
        vectorize: bool = True,
    ):
        super().__init__(func, objs, use_cache, clear_cache)
        self._vectorize = vectorize
    
    def load_arg(
        self, 
        directories: Iterable[str], 
        func : Callable[[Unpack[tuple]], Any] | None = None,
    ) -> Any | list[Any]:
        
        args_loaded = []
        for d in directories:
            arg_loaded = super().load_arg(d, self.id_func)
            args_loaded.append(arg_loaded)

        if func is None:
            func = self._func

        if self._vectorize:
            return [func(*i) if isinstance(i, tuple) else func(i) for i in args_loaded]
        else:
            return func(*args_loaded)


@overload
def co_proxy(
    obj: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]], 
    *,
    use_cache: bool = True,
    clear_cache: bool = False
) -> CoProxy:
    """
    `co_proxy('x') -> [x₀, x₁, ...]`
    """
    ...


@overload
def co_proxy(
    func: Callable[[Any], Any], 
    obj: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]], 
    *,
    use_cache: bool = True,
    clear_cache: bool = False,
    vectorize: bool = True,
) -> CoProxy:
    """
    If `vectorize` is `True` \\
    `co_proxy(f, 'x') -> [f(x₀), f(x₁), ...]` \\
    otherwise \\
    `co_proxy(f, 'x') -> f(x₀, x₁, ...)`
    """
    ...


@overload
def co_proxy(
    obj: list[tuple[ObjectName, ObjectType, FileName, Unpack[tuple]]], 
    *,
    use_cache: bool = True,
    clear_cache: bool = False,
) -> CoProxy:
    """
    `co_proxy(['x', 'y', ...]) -> [(x₀, y₀, ...), (x₁, y₁, ...), ...]`
    """
    ...


@overload
def co_proxy(
    func: Callable[..., Any], 
    obj: list[
        tuple[ObjectName, FileName, Unpack[tuple]]
        | tuple[ObjectName, ObjectType, FileName, Unpack[tuple]]
    ],
    *, 
    use_cache: bool = True,
    clear_cache: bool = False,
    vectorize: bool = True,
) -> CoProxy:
    """
    If `vectorize` is `True` \\
    `co_proxy(f, ['x', 'y', ...]) -> [f(x₀, y₀, ...), f(x₁, y₁, ...), ...]` \\
    otherwise \\
    `co_proxy(f, ['x', 'y', ...]) -> f((x₀, y₀, ...), (x₁, y₁, ...), ...]`
    """


@singledispatch
def co_proxy(
    func, 
    objs, 
    use_cache = True,
    clear_cache = False,
    vectorize = True,
):
    assert callable(func)
    if not isinstance(objs, list):
        objs = [objs]
    return CoProxy(func, objs, use_cache, clear_cache, vectorize)


@co_proxy.register(list)
def _(obj, **kwargs):
    return co_proxy(CoProxy.id_func, obj, **kwargs)


@co_proxy.register(tuple)
def _(obj, **kwargs):
    return co_proxy(CoProxy.id_func, obj, **kwargs)


