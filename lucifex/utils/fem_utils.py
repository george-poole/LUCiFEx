from enum import Enum

import numpy as np
from basix.ufl_wrapper import BasixElement
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, Constant, FunctionSpace, Expression
from ufl import Form
from ufl.core.expr import Expr
from ufl.finiteelement import FiniteElement
from ufl.algorithms.analysis import extract_coefficients, extract_constants


def is_scalar(
    f: Function | Constant | FunctionSpace | Expr,
) -> bool:
    """Returns `True` if the function or function space is scalar-valued."""
    return is_shape(f, ())


def is_vector(
    f: Function | Constant | FunctionSpace | Expr,
    dim: int | None = None,
) -> bool:
    """Returns `True` if the function or function space is vector-valued."""
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
        super().__init__(f'Shapes {u.ufl_shape} and {shape} do not match.')


class ScalarError(ShapeError):
    def __init__(self, u):
        super().__init__(u, "'scalar'")


class VectorError(ShapeError):
    def __init__(self, u):
        super().__init__(u, "'vector'")


class ScalarVectorError(ShapeError):
    def __init__(self, u):
        super().__init__(u, "'scalar or vector'")


def is_mixed_space(fs: FunctionSpace) -> bool:
    """e.g. Returns `True` if the function space is mixed.

    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ x DPₖ₋₁` -> `True`
    """
    return fs.num_sub_spaces > 0


def is_component_space(fs: FunctionSpace) -> bool:
    """Returns `True` if the function space is a mixed space 
    in which all subspaces are identical and scalar-valued.
    
    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ` -> `False`
    """
    if not is_mixed_space(fs):
        return False
    else:
        subspaces = function_subspaces(fs)
        sub0 = subspaces[0]
        if not is_scalar(sub0):
            return False
        return all([sub.ufl_element() == sub0.ufl_element() for sub in subspaces])
    

def function_subspace(
    fs: FunctionSpace,
    subspace_index: int | None = None,
    collapse: bool = True,
) -> FunctionSpace:
    if subspace_index is None:
        return fs
    fs_sub = fs.sub(subspace_index)
    if collapse:
        return fs_sub.collapse()[0]
    else:
        return fs_sub


def function_subspaces(
    fs: FunctionSpace,
    collapse: bool = True,
) -> tuple[FunctionSpace, ...]:
    subspaces = []
    n_sub = fs.num_sub_spaces
    for n in range(n_sub):
        subspaces.append(function_subspace(fs, n, collapse))
    return tuple(subspaces)


class ElementFamilyType(set, Enum):
    CONTINUOUS_LAGRANGE = {"P", "Q", "CG", "Lagrange"}
    DISCONTINOUS_LAGRANGE = {"DP", "DQ", "DG", "Discontinuous Lagrange"}
    LAGRANGE = CONTINUOUS_LAGRANGE.union(DISCONTINOUS_LAGRANGE)
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
) -> bool | None:
    if name == other:
        return True
    else:
        for family in ElementFamilyType:
            if (name in family) and (other in family):
                return True
            if (name in family) and (other not in family):
                return False
        return None
    

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


def extract_mesh(
    expr: Expr | Expression | Function,
) -> Mesh:
    meshes = extract_meshes(expr)
    if len(meshes) == 0:
        raise ValueError(
            "No mesh deduced from expression's operands."
        )
    if len(meshes) > 1:
        raise ValueError(
            "Multiple meshes deduced from expression's operands."
        )
    return list(meshes)[0]


def extract_meshes(
    expr: Expr | Expression,
) -> list[Mesh]:
    if isinstance(expr, Expression):
        return extract_meshes(expr.ufl_expression)

    meshes = set()
    coeffs_consts = (*extract_coefficients(expr), *extract_constants(expr))
    for c in coeffs_consts:
        if isinstance(c, Function):
            meshes.add(c.function_space.mesh)
        if hasattr(c, 'mesh'):
            meshes.add(getattr(c, 'mesh'))

    return meshes


def extract_integrands(
    form: Form,
) -> list[Expr]:
    return [i.integrand for i in form.integrals()]


def extract_integrand(
    form: Form,
) -> Expr:
    return sum(extract_integrands(form))
    
    