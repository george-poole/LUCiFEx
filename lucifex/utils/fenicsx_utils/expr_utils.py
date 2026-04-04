from typing import Any, TypeVar, TypeAlias

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, Constant, FunctionSpace, Expression
from ufl import Argument
from ufl.core.expr import Expr
from ufl.algorithms.analysis import extract_coefficients, extract_constants


T = TypeVar('T')
Scaled: TypeAlias = tuple[Constant | float, T]


def is_scalar(
    f: Function | Constant | FunctionSpace | Expr,
) -> bool:
    return is_shape(f, ())


def is_vector(
    f: Function | Constant | FunctionSpace | Expr,
    dim: int | None = None,
) -> bool:
    return is_shape(f, (dim,))


def is_tensor(f: Function | Constant | FunctionSpace | Expr):
    return not is_scalar(f) and not is_vector(f)


def is_shape(
    f: Function | Constant | FunctionSpace | Expr,
    shape: tuple[int | None, ...],
):
    if isinstance(f, (tuple, list, float, int, np.ndarray)):
        _shape = np.array(f).shape
    elif isinstance(f, FunctionSpace):
        _shape = f.ufl_element().value_shape()
    else:
        _shape = f.ufl_shape

    if len(shape) != len(_shape):
        return False
    else:
        shape = tuple(
            i if i is not None else j for i, j in zip(shape, _shape, strict=True)
        )
        return shape == _shape
    

class ShapeError(ValueError):
    def __init__(
        self,
        u: Function | Constant | Expr, 
        shape: str | tuple,
    ):
        """
        Error to raise if shape does not match.
        """
        super().__init__(f'Shapes {u.ufl_shape} and {shape} do not match.')


class NonScalarError(ShapeError):
    def __init__(self, u):
        """
        Error to raise if non-scalar provided.
        """
        super().__init__(u, "'scalar'")


class NonVectorError(ShapeError):
    def __init__(self, u):
        """
        Error to raise if non-vector provided.
        """
        super().__init__(u, "'vector'")


class NonScalarVectorError(ShapeError):
    def __init__(self, u):
        """
        Error to raise if tensor provided.
        """
        super().__init__(u, "'scalar or vector'")


class MeshExtractionError(ValueError):
    def __init__(self, u, unique: bool = False):
        """
        Error to raise if no mesh could be extracted.
        """
        msg = ' a unique 'if unique else ' '
        super().__init__(f'Could not extract{msg}mesh from {u}.')


def extract_mesh(
    expr: Expr | Expression | Function | Any,
) -> Mesh:
    if isinstance(expr, Function):
        return expr.function_space.mesh
    meshes = extract_meshes(expr)
    if len(meshes) == 0:
        raise MeshExtractionError(expr)
    if len(meshes) > 1:
        raise MeshExtractionError(expr, unique=True)
    return list(meshes)[0]


def extract_meshes(
    expr: Expr | Expression | Any,
) -> list[Mesh]:
    if isinstance(expr, Expression):
        return extract_meshes(expr.ufl_expression)

    meshes = set()
    coeffs_consts = (
        *extract_coefficients(expr), 
        *extract_constants(expr),
    )
    for c in coeffs_consts:
        if isinstance(c, Function):
            meshes.add(c.function_space.mesh)
        if hasattr(c, 'mesh'):
            meshes.add(getattr(c, 'mesh'))

    return meshes


def extract_function_space(
    arg: Function | Argument | FunctionSpace | Any,
) -> FunctionSpace:
    if isinstance(arg, Function):
        fs = arg.function_space
    elif isinstance(arg, Argument):
        fs = arg.ufl_function_space()
    elif isinstance(arg, FunctionSpace):
        fs = arg
    else:
        fs = getattr(arg, 'function_space')

    return fs