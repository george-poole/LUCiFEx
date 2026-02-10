from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
from types import MappingProxyType
from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    overload,
    TypeAlias,
    Generic,
    Any,
)
from collections.abc import (
    Iterable,
    Hashable,
)
from functools import lru_cache, update_wrapper, wraps
from inspect import signature
import time


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return repr(self.value)


class FloatEnum(float, Enum):
    def __str__(self) -> str:
        return repr(self.value)


class classproperty:
    def __init__(self, func):
        self._getfunc = func

    def __get__(self, _, owner):
        return self._getfunc(owner)

            
class ToDoError(NotImplementedError):
    def __init__(self):
        super().__init__('Working on it! Coming soon...')