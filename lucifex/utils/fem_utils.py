from enum import Enum

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
    if isinstance(f, FunctionSpace):
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
    

def ShapeError(
    u: Function | Constant | Expr, 
    shape: str | tuple,
):
    return ValueError(f'Expected {shape}, not shape {u.ufl_shape}.')


def VectorError(u):
    return ShapeError(u, 'scalar')


def VectorError(u):
    return ShapeError(u, 'vector')


def ScalarVectorError(u):
    return ShapeError(u, 'scalar or vector')


def is_mixed_space(function_space: FunctionSpace) -> bool:
    """e.g. Returns `True` if the function space is mixed.

    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ x DPₖ₋₁` -> `True`
    """
    return function_space.num_sub_spaces > 0


def is_component_space(function_space: FunctionSpace) -> bool:
    """Returns `True` if the function space is a mixed space 
    in which all subspaces are identical and scalar-valued.
    
    `Pₖ × ... × Pₖ` -> `True` \\
    `BDMₖ` -> `False`
    """
    if not is_mixed_space(function_space):
        return False
    else:
        subspaces = extract_subspaces(function_space)
        sub0 = subspaces[0]
        if not is_scalar(sub0):
            return False
        return all([sub.ufl_element() == sub0.ufl_element() for sub in subspaces])


def extract_subspaces(
    function_space: FunctionSpace,
    collapse: bool = True,
) -> tuple[FunctionSpace, ...]:
    subspaces = []
    n_sub = function_space.num_sub_spaces
    for n in range(n_sub):
        if collapse:
            sub, _ = function_space.sub(n).collapse()
        else:
            sub = function_space.sub(n)
        subspaces.append(sub)
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


def subspace_functions(
    f: Function,
    collapse: bool = True,
    # TODO copy: bool = False,
) -> tuple[Function, ...]:
    if collapse:
        return tuple(i.collapse() for i in f.split())
    else:
        return tuple(i for i in f.split())


def extract_mesh(
    expr: Expr | Expression,
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
    
    