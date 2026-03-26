from enum import Enum
from typing import Any, TypeVar, TypeAlias, Callable, Iterable
from inspect import Signature, Parameter

import numpy as np
from basix.ufl_wrapper import BasixElement
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, Constant, FunctionSpace, Expression
from ufl import Argument, replace
from ufl.core.expr import Expr
from ufl.finiteelement import FiniteElement
from ufl.algorithms.analysis import extract_coefficients, extract_constants, extract_arguments


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


def extract_expr_factory(
    expr: Expr,
    strict: bool = True,
    forge: bool | Iterable[str] = False,
    extractors: Iterable[Callable[[Expr], Iterable]] = (
        extract_arguments,
        extract_coefficients,
        extract_constants,
    )
) -> Callable[..., Expr | Any]:
    extracted = []
    for ext in extractors:
        extracted.extend(ext(expr))

    def _(*args):
        mapping = dict(zip(extracted, args, strict=strict))
        return replace(expr, mapping)
    
    if forge:
        if forge is True:
            names = [getattr(c, 'name') for c in extracted]
        else:
            names = forge
        paramspec_params = [
            Parameter(n, Parameter.POSITIONAL_OR_KEYWORD, annotation=type(c)) 
            for n, c in zip(names, extracted, strict=True)
        ]
        _.__signature__ = Signature(
            paramspec_params, 
            return_annotation=bool,
        )

    return _


class ElementFamilyType(set, Enum):
    CONTINUOUS_LAGRANGE = {"P", "Q", "CG", "Lagrange"}
    DISCONTINOUS_LAGRANGE = {"DP", "DQ", "DG", "Discontinuous Lagrange"} 
    LAGRANGE = DISCONTINOUS_LAGRANGE | CONTINUOUS_LAGRANGE
    BREZZI_DOUGLAS_MARINI = {"BDM", "Brezzi-Douglas-Marini"}
    RAVIART_THOMAS = {"RT", "Raviart-Thomas"}


def is_same_element(
    f: Function | FunctionSpace,
    family: str,
    degree: int | None = None,
    dim: int | None = None,
    mesh: Mesh | None = None,
) -> bool:
    if isinstance(f, Function):
        f = f.function_space

    f_element: BasixElement | FiniteElement = f.ufl_element()
    f_fam = f_element.family()
    f_deg = f_element.degree()
    f_shape = f_element.value_shape()
    f_mesh = f.mesh
    if isinstance(f_element, BasixElement):
        f_dc = f_element.discontinuous
    else:
        f_dc = is_discontinuous_family(f_fam)

    if degree is None:
        degree = f_deg
    if dim is None:
        shape = f_shape
    else:
        shape = (dim, )
    if mesh is None:
        mesh = f_mesh

    if (is_family_alias(f_fam, family) is True 
        and f_deg == degree 
        and f_mesh == mesh
        and f_shape == shape
        and f_dc == is_discontinuous_family(family)):
        return True
    else:
        return False
    

def is_family_alias(
    name: str,
    other: str,
) -> bool:
    if name == other:
        return True
    else:
        for family in ElementFamilyType:
            if (name in family) and (other in family):
                return True
            # if (name in family) and (other not in family):
            #     return False
        return False
    

def is_continuous_lagrange(
    f: Function | FunctionSpace,
    degree: int | None = None,
) -> bool:
    return is_same_element(f, "P", degree)


def is_discontinuous_lagrange(
    f: Function | FunctionSpace,
    degree: int | None = None,
) -> bool:
    return is_same_element(f, "DP", degree)


def is_discontinuous_family(family: str) -> bool:
    return family in ElementFamilyType.DISCONTINOUS_LAGRANGE or family.startswith(
        "Discontinuous"
    )

