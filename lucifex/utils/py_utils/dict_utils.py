from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    TypeVar,
    overload,
    Generic,
    Any,
)
from typing_extensions import Self


K = TypeVar('K')
V = TypeVar('V')
class MultiKey(ABC, Generic[K, V]):
    """
    Abstract base class for types with multi-key access
    """
    @overload
    def __getitem__(
        self, 
        key: K,
    ) -> V:
        ...

    @overload
    def __getitem__(
        self, 
        keys: tuple[K, ...],
    ) -> tuple[V, ...]:
        ...

    def __getitem__(self, key):
        if isinstance(key, tuple) and all(isinstance(i, type(key[0])) for i in key):
            return tuple(self[i] for i in key)
        else:
            return self._getitem(key)
        
    @abstractmethod
    def _getitem(
        self,
        key: K,
    ) -> V:
        ...


K = TypeVar('K')
V = TypeVar('V')
class FrozenDict(MultiKey[K, V], Generic[K, V]):
    """
    An immutable dictionary with multi-key access
    """

    @overload
    def __init__(self: 'FrozenDict[K]', d: dict[K, V]):
        ...

    @overload
    def __init__(self: 'FrozenDict[str, V]', **kwargs: V):
        ...

    def __init__(self, *args: dict, **kwargs):
        if kwargs and args:
            raise TypeError
        if kwargs and not args:
            self._dict = kwargs
        if args and not kwargs:
            if len(args) > 1:
                raise TypeError
            self._dict = args[0].copy()

    def _getitem(
        self,
        key: str,
    ):
        return self._dict[key]

    def __str__(self) -> str:
        return str(self._dict)
    
    def __repr__(self) -> str:
        kws = ', '.join([f'{k}={v}' for k, v in self._dict.items()])
        return f'{self.__class__.__name__}({kws})'
    
    def replace(self, **kwargs: Any) -> Self:
        _kwargs = self._dict.copy()
        _kwargs.update(kwargs)
        return self.__class__(**_kwargs)
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()
        


@overload
def nested_dict(
    *,
    depth: int | None = None,
) -> defaultdict | dict:
    ...

K = TypeVar('K')
V = TypeVar('V')
@overload
def nested_dict(
    _: tuple[type[K], type[V]],
    /,
    *,
    depth: int | None = None,
) -> dict[K, V]:
    ...

K1 = TypeVar('K1')
K2 = TypeVar('K2')
V = TypeVar('V')
@overload
def nested_dict(
    _: tuple[type[K1], type[K2], type[V]],
    /,
    *,
    depth: int | None = None,
) -> dict[K1, dict[K2, V]]:
    ...

K1 = TypeVar('K1')
K2 = TypeVar('K2')
K3 = TypeVar('K3')
V = TypeVar('V')
@overload
def nested_dict(
    _: tuple[type[K1], type[K2], type[K3], type[V]],
    /,
    *,
    depth: int | None = None,
) -> dict[K1, dict[K2, dict[K3, V]]]:
    ...


def nested_dict(
    _: tuple | None = None,
    /,
    *,
    depth: int | None = None,
):
    if depth is None and _ is None:
        return defaultdict(nested_dict)
    if depth is None and _ is not None:
        depth = len(_) - 1

    if depth <= 0:
        raise ValueError('Depth must be a non-negative integer.')
    elif depth == 1:
        return dict()
    elif depth == 2:
        return defaultdict(dict)
    else:
        return defaultdict(lambda: nested_dict(depth=depth - 1))

