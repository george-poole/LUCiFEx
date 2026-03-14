from collections.abc import Iterable
from functools import singledispatch

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.fem import Constant

from ..py_utils import MultipleDispatchTypeError
from .expr_utils import ShapeError


def create_constant(
    mesh: Mesh,
    value: float | Iterable[float] | Constant,
    try_identity: bool = False,
) -> Constant:
    """
    Typecast to `dolfinx.fem.Constant` 
    """
    if try_identity and isinstance(value, Constant) and value.ufl_domain() is mesh.ufl_domain():
        return value
    else:
        return _create_constant(value, mesh)


@singledispatch
def _create_constant(value, _):
    raise MultipleDispatchTypeError(value)


@_create_constant.register(float)
@_create_constant.register(int)
def _(value, mesh: Mesh,):
    return Constant(mesh, float(value))


@_create_constant.register(Iterable)
def _(value: Iterable[float], mesh: Mesh):
    if all(isinstance(i, (float, int)) for i in value):
        value = [float(i) for i in value]
        return Constant(mesh, value)
    else:
        raise TypeError('Expected an iterable of numbers')


def set_constant(
    c: Constant,
    value: Constant | float | np.ndarray | Iterable[float],
) -> None:
    """
    Mutates `c` by setting its value array. Does not mutate `value`.
    """
    return _set_constant(value, c)


@singledispatch
def _set_constant(value, *_, **__):
    raise MultipleDispatchTypeError(value)


@_set_constant.register(Constant)
def _(value: Constant, const: Constant):
    const.value = value.value.copy()


@_set_constant.register(float)
@_set_constant.register(int)
def _(value, const: Constant):
    const.value = value

@_set_constant.register(np.ndarray)
def _(value: np.ndarray, const: Constant):
    if not const.value.shape == value.shape:
        raise ShapeError(const, value.shape)
    const.value = value

@_set_constant.register(Iterable)
def _(value, const: Constant):
    return _set_constant(np.array(value), const)



