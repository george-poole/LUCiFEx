from collections.abc import Callable, Iterable
from functools import singledispatch
from typing_extensions import Self

import numpy as np
from ufl.core.expr import Expr
from ufl.algorithms.analysis import extract_coefficients, extract_constants
from dolfinx.fem import Function, Constant

from ..utils.py_utils import MultipleDispatchTypeError


class UnsolvedType:
    """Singleton object representing an unsolved value"""
    value: float = np.nan
    is_unsolved = staticmethod(np.isnan)

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnsolvedType, cls).__new__(cls)
        return cls._instance
    
    def __float__(self) -> float:
        return self.value
    
    def __repr__(self) -> str:
        return 'Unsolved'

    def __add__(self, _) -> Self:
        return self

    def __add__(self, _) -> Self:
        return self

    def __sub__(self, _) -> Self:
        return self

    def __mul__(self, _) -> Self:
        return self

    def __rmul__(self, _) -> Self:
        return self

    def __div__(self, _) -> Self:
        return self


Unsolved = UnsolvedType()


def is_unsolved(
    obj: Function | Constant | Expr,
    iter_func: Callable[[Iterable], bool] = np.any,
) -> bool:
    return _is_unsolved(obj, iter_func)


@singledispatch
def _is_unsolved(obj, _):
    raise MultipleDispatchTypeError(obj)


@_is_unsolved.register(Function)
def _(obj: Function, iter_func):
    return iter_func(UnsolvedType.is_unsolved(obj.x.array))


@_is_unsolved.register(Constant)
def _(obj: Constant, iter_func):
    if obj.value.shape == ():
        return UnsolvedType.is_unsolved(obj.value)
    else:
        return iter_func(UnsolvedType.is_unsolved(obj.value))


@_is_unsolved.register(Expr)
def _(obj: Expr, iter_func):
    coeffs_consts = (*extract_coefficients(obj), *extract_constants(obj))
    return iter_func([_is_unsolved(i, iter_func) for i in coeffs_consts])