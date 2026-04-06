from enum import Enum

from basix.ufl_wrapper import BasixElement
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, FunctionSpace
from ufl.finiteelement import FiniteElement


class ElementFamilyType(set, Enum):
    CONTINUOUS_LAGRANGE = {"P", "Q", "CG", "Lagrange"}
    DISCONTINOUS_LAGRANGE = {"DP", "DQ", "DG", "Discontinuous Lagrange"} 
    LAGRANGE = DISCONTINOUS_LAGRANGE | CONTINUOUS_LAGRANGE
    BREZZI_DOUGLAS_MARINI = {"BDM", "Brezzi-Douglas-Marini"}
    RAVIART_THOMAS = {"RT", "Raviart-Thomas"}


def is_equivalent_element(
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

    if (is_equivalent_family(f_fam, family) is True 
        and f_deg == degree 
        and f_mesh == mesh
        and f_shape == shape
        and f_dc == is_discontinuous_family(family)):
        return True
    else:
        return False
    

def is_equivalent_family(
    name: str,
    other: str,
) -> bool:
    if name == other:
        return True
    else:
        for family in ElementFamilyType:
            if (name in family) and (other in family):
                return True
        return False
    

def is_continuous_lagrange(
    f: Function | FunctionSpace,
    degree: int | None = None,
) -> bool:
    return is_equivalent_element(f, "P", degree)


def is_discontinuous_lagrange(
    f: Function | FunctionSpace,
    degree: int | None = None,
) -> bool:
    return is_equivalent_element(f, "DP", degree)


def is_discontinuous_family(family: str) -> bool:
    return family in ElementFamilyType.DISCONTINOUS_LAGRANGE or family.startswith(
        "Discontinuous"
    )

