from collections.abc import Callable, Iterable
from functools import singledispatch
from typing_extensions import Self

import numpy as np
from ufl import Form
from ufl.core.expr import Expr
from ufl.algorithms.analysis import extract_coefficients, extract_constants
from dolfinx.fem import Function, Constant

from ..utils.py_utils import OverloadTypeError


class UnsolvedType:
    value: float = np.nan
    is_equal = staticmethod(np.isnan)

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
"""
Singleton representing a quantity that has not yet been solved for.
"""


def is_unsolved(
    obj: Function | Constant | Expr | Form,
    iter_func: Callable[[Iterable], bool] = np.any,
) -> bool:
    return _is_unsolved(obj, iter_func)


@singledispatch
def _is_unsolved(obj, _):
    raise OverloadTypeError(obj)


@_is_unsolved.register(Function)
def _(obj: Function, iter_func):
    return iter_func(UnsolvedType.is_equal(obj.x.array))


@_is_unsolved.register(Constant)
def _(obj: Constant, iter_func):
    if obj.value.shape == ():
        return UnsolvedType.is_equal(obj.value)
    else:
        return iter_func(UnsolvedType.is_equal(obj.value))


@_is_unsolved.register(Expr)
def _(obj: Expr, iter_func):
    coeffs_consts = (*extract_coefficients(obj), *extract_constants(obj))
    return iter_func([_is_unsolved(i, iter_func) for i in coeffs_consts])


@_is_unsolved.register(Form)
def _(form: Form, iter_func):
    for i in [*form.coefficients(), *form.constants()]:
        if is_unsolved(i, iter_func):
            return True
    return False