from typing import Literal

from ufl.core.expr import Expr
from ufl import Form, Measure, conditional, gt, FacetNormal, Argument
from dolfinx.mesh import Mesh

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, ExprSeries,
    FiniteDifference, FiniteDifferenceArgwise, 
    FiniteDifferenceDerivative, DT, AB1,
)
from lucifex.fdm.ufl_operators import inner, grad, div
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import is_none


def derivative_form(
    v: Argument,
    u: Function | FunctionSeries,
    dt: ConstantSeries | Constant,
    D_dt: FiniteDifferenceDerivative = DT,
    dx: Measure | Expr | Literal[1] = 1,
) -> Form | Expr:
    """    
    `∫dx v∂u/∂t`
    """
    return v * D_dt(u, dt) * dx


def advection_forms(
    v: Argument,
    u: Function | FunctionSeries,
    a,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = AB1,
    bcs: BoundaryConditions | tuple | None = None,
    dx: Measure | Expr | Literal[1] = 1,
    *,
    by_parts: bool = False,
) -> list[Form | Expr]:
    """
    `∫dx v(𝐚·∇u)`
    """

    forms = []

    if not by_parts:
        # TODO or simply `v * inner(a, grad(u))`
        adv = v * ExprSeries(inner(a, grad(u)), args=(a, u)) 
    else:
        adv = ExprSeries(-inner(grad(v), a * u), args=(a, u))
    
    F_adv = D_adv(adv, trial=u) * dx
    forms.append(F_adv)

    if by_parts and (bcs is not None):
        ds, u_dirichlet = (
            bcs.boundary_data(u, 'dirichlet') if isinstance(bcs, BoundaryConditions)
            else bcs
        )
        n = FacetNormal(u.function_space.mesh)
        lmbda = conditional(gt(inner(n, a), 0), 1, 0)
        F_ds = v * lmbda * inner(n, a) * u * ds
        F_ds += sum([v * (1 - lmbda) * inner(n, a) * uD * ds(i) for i, uD in u_dirichlet])
        forms.append(F_ds)

    return forms


def diffusion_forms(
    v: Argument,
    u: Function | FunctionSeries,
    d,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = AB1,
    bcs: BoundaryConditions | None = None,
    dx: Measure | Expr | Literal[1] = 1,
    *,
    by_parts: bool = True,
) -> list[Form | Expr]:
    """
    `∫dx v∇·(D·∇u) = - ∫dx ... + ∫dS ... `
    """
    if not by_parts:
        diff = v * ExprSeries(div(d * grad(u)), args=(d, u)) 
    else:
        diff = -inner(grad(v), ExprSeries(d * grad(u), args=(d, u)))

    F_diff = D_diff(diff, trial=u) * dx 
    forms = [F_diff]  

    if by_parts and (bcs is not None):
        ds, u_neumann, u_robin = bcs.boundary_data(u, 'neumann', 'robin')
        if u_neumann:
            F_neumann = sum([v * uN * ds(i) for i, uN in u_neumann])
            forms.append(F_neumann)
        if u_robin:
            F_robin = sum([v * uR * ds(i) for i, uR in u_robin])
            forms.append(F_robin)

    return forms


def reaction_forms(
    v: Argument,
    u: Function | FunctionSeries,
    r: Function | Constant | Expr | None = None,
    j: Function | Constant | Expr | None = None,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    dx: Measure | Expr | Literal[1] = 1,
) -> list[Form | Expr]:
    """
    `vRu + vJ`
    """
    forms = [] 
    if not is_none(r):
        # TODO or simply `r * u` with args deduced
        reac = ExprSeries(r * u, args=(r, u))
        F_reac = v * D_reac(reac, trial=u) * dx
        forms.append(F_reac)
    if not is_none(j):
        F_src = v * D_src(j, trial=u) * dx
        forms.append(F_src)
    return forms


class OptionError(ValueError):
    def __init__(self, opt: int):
        super().__init__(f"Invalid option number '{opt}'")