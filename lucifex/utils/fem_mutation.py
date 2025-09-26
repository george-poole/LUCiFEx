from typing import Any
from collections.abc import Callable, Iterable
from functools import singledispatch

from dolfinx.fem import Function, Constant, Function
from dolfinx.fem import Expression
from petsc4py import PETSc
from ufl.core.expr import Expr
import numpy as np

from .fem_utils import is_scalar
from .py_utils import StrSlice, as_slice, MultipleDispatchTypeError


# TODO @overload
def set_fem_function(
    f: Function,
    value: Function | Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float],
    dofs_indices: Iterable[int] | StrSlice | None = None,
) -> None:
    """Mutates `f`, does not mutate `value`"""
    if isinstance(dofs_indices, StrSlice):
        dofs_indices = as_slice(dofs_indices)
    elif isinstance(dofs_indices, Iterable):
        dofs_indices = np.array(dofs_indices), np.asarray
    return _set_fem_function(value, f, dofs_indices)


@singledispatch
def _set_fem_function(value):
    raise MultipleDispatchTypeError(value)


@_set_fem_function.register(Expression)
@_set_fem_function.register(Expr)
@_set_fem_function.register(Callable)
def _(value, f: Function, indices):
    assert indices is None
    interpolate_fem_function(f, value)


@_set_fem_function.register(Function)
def _(value: Function, f: Function, indices: np.ndarray | None):
    if indices is None:
        f.interpolate(value)
    else:
        assert f.function_space == value.function_space
        f.x.array[indices] = value.x.array[indices]


@_set_fem_function.register(Constant)
def _(value: Constant, f: Function, indices: np.ndarray | None):
    if value.value.shape == ():
        _set_fem_function(value.value.item(), f, indices)
    else:
        assert indices is None
        assert value.value.shape == f.ufl_shape
        interpolate_fem_function(f, value)


@_set_fem_function.register(float)
@_set_fem_function.register(int)
def _(value, u: Function, indices: np.ndarray | None):
    if indices is None:
        interpolate_fem_function(u, value)
    else:
        u.x.array[indices] = value

@_set_fem_function.register(Iterable)
def _(value, u: Function, indices: np.ndarray | None):
    if indices is None:
        interpolate_fem_function(u, value)
    else:
        u.x.array[indices] = value[indices]


def interpolate_fem_function(
    f: Function,
    value: Function | Constant | Callable[[np.ndarray], np.ndarray] | Expression | Expr | float | Iterable[float | Callable[[np.ndarray], np.ndarray]],
) -> None:
    """Mutates `f` by calling its `interpolate` method"""
    return _interpolate_fem_function(value, f)


@singledispatch
def _interpolate_fem_function(u, *_):
    raise MultipleDispatchTypeError(u)


@_interpolate_fem_function.register(Expr)
def _(u: Expr, f: Function):
    f.interpolate(Expression(u, f.function_space.element.interpolation_points()))

@_interpolate_fem_function.register(Function)
@_interpolate_fem_function.register(Expression)
@_interpolate_fem_function.register(Callable)
def _(u, f: Function):
    f.interpolate(u)

@_interpolate_fem_function.register(Constant)
def _(u: Constant, f: Function):
    if is_scalar(u):
        return _interpolate_fem_function(u.value.item(), f)
    else:
        return _interpolate_fem_function(u.value, f)
    

@_interpolate_fem_function.register(Iterable)
def _(u: Iterable[float | Callable], f: Function):
    def _lambda_x(u) -> Callable[[np.ndarray], np.ndarray]:
        if isinstance(u, (int ,float)):
            return lambda x: np.full_like(x[0], u, dtype=PETSc.ScalarType)
        if isinstance(u, Constant):
            assert is_scalar(u)
            return _lambda_x(u.value.item())
        if isinstance(u, Callable):
            return u
        raise MultipleDispatchTypeError(u)
    f.interpolate(lambda x: np.vstack([_lambda_x(ui)(x) for ui in u]))


@_interpolate_fem_function.register(float)
@_interpolate_fem_function.register(int)
def _(u, f: Function):
    f.interpolate(lambda x: np.full_like(x[0], u, dtype=PETSc.ScalarType))


def set_fem_constant(
    c: Constant,
    value: Constant | float | np.ndarray | Iterable[float],
) -> None:
    """Mutates `c`"""
    return _set_fem_constant(value, c)


@singledispatch
def _set_fem_constant(value):
    raise MultipleDispatchTypeError(value)


@_set_fem_constant.register(Constant)
def _(value: Constant, const: Constant):
    const.value = value.value.copy()


@_set_fem_constant.register(float)
@_set_fem_constant.register(int)
def _(value, const: Constant):
    const.value = value

@_set_fem_constant.register(np.ndarray)
def _(value: np.ndarray, const: Constant):
    assert const.value.shape == value.shape
    const.value = value

@_set_fem_constant.register(Iterable)
def _(value, const: Constant):
    return _set_fem_constant(np.array(value), const)

def set_value(obj: Function | Constant, value: Any) -> None:
    return _set_value(obj, value)


@singledispatch
def _set_value(obj, _):
    raise MultipleDispatchTypeError(obj)


@_set_value.register(Constant)
def _(obj, value):
    return set_fem_constant(obj, value)


@_set_value.register(Function)
def _(obj, value):
    return set_fem_function(obj, value)