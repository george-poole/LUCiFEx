from abc import ABC, abstractmethod
from collections import defaultdict
from types import MappingProxyType
from typing import (
    TypeVar,
    overload,
    Generic,
    Any,
)
from functools import lru_cache, update_wrapper, wraps
from inspect import signature


T = TypeVar('T')
class MultiKey(ABC, Generic[T]):
    """
    Abstract base class for types with multi-key access
    """
    @overload
    def __getitem__(
        self, 
        key: str,
    ) -> T:
        ...

    @overload
    def __getitem__(
        self, 
        keys: tuple[str, ...],
    ) -> tuple[T, ...]:
        ...

    def __getitem__(
        self, 
        key: str | tuple[str, ...],
    ):
        if isinstance(key, tuple):
            return tuple(self[i] for i in key)
        elif isinstance(key, str):
            try:
                return self._getitem(key)
            except KeyError:
                # (f"'{key}' not found in simulation's namespace."
                raise KeyError
        else:
            raise TypeError
        
    @abstractmethod
    def _getitem(
        self,
        key: str,
    ) -> T:
        ...


class FrozenDict(MultiKey[Any]):
    """
    An immutable dictionary with multi-key access
    """

    def __init__(self, **kwargs: Any):
        self._dict = kwargs
        
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
        

def multi_dict(
    **kwargs: Any,
) -> MultiKey[Any]:
    
    _getitem = lambda k: kwargs[k]
    
    cls = type(
        'MultiDict',
        (MultiKey, ),
        {'_getitem', _getitem},
    )

    return cls()


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

    if depth == 1:
        return dict()
    if depth == 2:
        return defaultdict(dict)
    assert depth > 2
    return nested_dict(nested_dict(depth=depth - 1))

