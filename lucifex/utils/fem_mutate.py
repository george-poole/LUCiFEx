from typing import Any
from collections.abc import Callable, Iterable
from functools import singledispatch

from dolfinx.fem import Function, Constant, Function
from dolfinx.fem import Expression
from petsc4py import PETSc
from ufl.core.expr import Expr
import numpy as np

from .fem_utils import is_scalar, ShapeError
from .py_utils import StrSlice, as_slice, MultipleDispatchTypeError


# TODO @overload
def set_finite_element_function(
    f: Function,
    value: Function | Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float | Constant | Callable[[np.ndarray], np.ndarray]],
    dofs_indices: Iterable[int] | StrSlice | None = None,
) -> None:
    """
    Mutates `f`, does not mutate `value`.
    """
    if isinstance(dofs_indices, StrSlice):
        dofs_indices = as_slice(dofs_indices)
    elif isinstance(dofs_indices, Iterable):
        dofs_indices = np.array(dofs_indices)
    return _set_finite_element_function(value, f, dofs_indices)


@singledispatch
def _set_finite_element_function(value, *_, **__):
    raise MultipleDispatchTypeError(value)


@_set_finite_element_function.register(Expression)
@_set_finite_element_function.register(Expr)
@_set_finite_element_function.register(Callable)
def _(value, f: Function, indices):
    assert indices is None
    interpolate_finite_element_function(f, value)


@_set_finite_element_function.register(Function)
def _(value: Function, f: Function, indices: np.ndarray | None):
    if indices is None:
        f.interpolate(value)
    else:
        assert f.function_space == value.function_space
        f.x.array[indices] = value.x.array[indices]


@_set_finite_element_function.register(Constant)
def _(value: Constant, f: Function, indices: np.ndarray | None):
    if value.value.shape == ():
        _set_finite_element_function(value.value.item(), f, indices)
    else:
        assert indices is None
        if not f.ufl_shape == value.value.shape:
            raise ShapeError(f, value.value.shape)
        interpolate_finite_element_function(f, value)


@_set_finite_element_function.register(float)
@_set_finite_element_function.register(int)
def _(value, u: Function, indices: np.ndarray | None):
    if indices is None:
        interpolate_finite_element_function(u, value)
    else:
        u.x.array[indices] = value

@_set_finite_element_function.register(Iterable)
def _(value, u: Function, indices: np.ndarray | None):
    if indices is None:
        interpolate_finite_element_function(u, value)
    else:
        u.x.array[indices] = value[indices]


def interpolate_finite_element_function(
    f: Function,
    value: Function | Callable[[np.ndarray], np.ndarray] | Expression | Expr | Constant | float | Iterable[float | Constant | Callable[[np.ndarray], np.ndarray]],
) -> None:
    """
    Mutates `f` by calling its `interpolate` method, does not mutate `value`.
    """
    return _interpolate_finite_element_function(value, f)


@singledispatch
def _interpolate_finite_element_function(u, *_, **__):
    raise MultipleDispatchTypeError(u)


@_interpolate_finite_element_function.register(Expr)
def _(u: Expr, f: Function):
    f.interpolate(Expression(u, f.function_space.element.interpolation_points()))

@_interpolate_finite_element_function.register(Function)
@_interpolate_finite_element_function.register(Expression)
@_interpolate_finite_element_function.register(Callable)
def _(u, f: Function):
    f.interpolate(u)

@_interpolate_finite_element_function.register(Constant)
def _(u: Constant, f: Function):
    if is_scalar(u):
        return _interpolate_finite_element_function(u.value.item(), f)
    else:
        return _interpolate_finite_element_function(u.value, f)
    

@_interpolate_finite_element_function.register(Iterable)
def _(u: Iterable[float | Constant | Callable], f: Function):
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


@_interpolate_finite_element_function.register(float)
@_interpolate_finite_element_function.register(int)
def _(u, f: Function):
    f.interpolate(lambda x: np.full_like(x[0], u, dtype=PETSc.ScalarType))


def set_fem_constant(
    c: Constant,
    value: Constant | float | np.ndarray | Iterable[float],
) -> None:
    """
    Mutates `c`, does not mutate `value`.
    """
    return _set_finite_element_constant(value, c)


@singledispatch
def _set_finite_element_constant(value, *_, **__):
    raise MultipleDispatchTypeError(value)


@_set_finite_element_constant.register(Constant)
def _(value: Constant, const: Constant):
    const.value = value.value.copy()


@_set_finite_element_constant.register(float)
@_set_finite_element_constant.register(int)
def _(value, const: Constant):
    const.value = value

@_set_finite_element_constant.register(np.ndarray)
def _(value: np.ndarray, const: Constant):
    if not const.value.shape == value.shape:
        raise ShapeError(const, value.shape)
    const.value = value

@_set_finite_element_constant.register(Iterable)
def _(value, const: Constant):
    return _set_finite_element_constant(np.array(value), const)

def set_value(obj: Function | Constant, value: Any) -> None:
    return _set_value(obj, value)


@singledispatch
def _set_value(obj, *_, **__):
    raise MultipleDispatchTypeError(obj)


@_set_value.register(Constant)
def _(obj, value):
    return set_fem_constant(obj, value)


@_set_value.register(Function)
def _(obj, value):
    return set_finite_element_function(obj, value)