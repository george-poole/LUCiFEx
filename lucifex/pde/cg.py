from typing import Literal

from ufl.core.expr import Expr
from ufl import Form, Measure

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, ExprSeries,
    FiniteDifference, FiniteDifferenceArgwise, 
    FiniteDifferenceDerivative, DT, AB1,
)
from lucifex.fdm.ufl_operators import inner, grad, div
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import is_zero


def derivative_form(
    v,
    u,
    dt: ConstantSeries | Constant,
    D_dt: FiniteDifferenceDerivative = DT,
    dx: Measure | Expr | Literal[1] = 1,
) -> Form | Expr:
    """    
    `∫dx v∂u/∂t`
    """
    return v * D_dt(u, dt) * dx


def advection_form(
    v,
    u,
    a,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = AB1,
    dx: Measure | Expr | Literal[1] = 1,
) -> Form | Expr:
    """
    `∫dx v(𝐚·∇u)`
    """

    expr = lambda a, u: inner(a, grad(u))
    if isinstance(D_adv, FiniteDifference):
        adv = D_adv(expr(a, u), trial=u)
    else:
        adv = D_adv(ExprSeries(expr)(a, u), trial=u)
        # adv = D_adv(expr, a, u, trial=u)
    F_adv =  v * adv * dx
    return F_adv


def diffusion_forms(
    v,
    u: Function | FunctionSeries,
    d,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = AB1,
    bcs: BoundaryConditions | None = None,
    dx: Measure | Expr | Literal[1] = 1,
    divergence: bool = False,
) -> list[Form]:
    """
    `∫dx v∇·(D·∇u) = - ∫dx ... + ∫dS ... `
    """
    if divergence:
        F_diff = v * div(d * grad(D_diff(u))) * dx 
    else:
        expr = lambda d, u: d * grad(u)
        if isinstance(D_diff, FiniteDifference):
            diff = D_diff(expr(d, u), trial=u)
        else:
            diff = D_diff(expr, d, u, trial=u)
        F_diff = -inner(grad(v), diff) * dx    
   
    forms = [F_diff]

    if bcs is not None:
        ds, u_neumann, u_robin = bcs.boundary_data(u, 'neumann', 'robin')
        if u_neumann:
            F_neumann = sum([v * uN * ds(i) for i, uN in u_neumann])
            forms.append(F_neumann)
        if u_robin:
            F_robin = sum([v * uR * ds(i) for i, uR in u_robin])
            forms.append(F_robin)

    return forms


def reaction_forms(
    v,
    u: Function,
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
    if not is_zero(r):
        _expr = lambda r, u: r * u
        if isinstance(D_reac, FiniteDifference):
            expr = D_reac(_expr(r, u), trial=u)
        else:
            expr = D_reac(_expr, r, u, trial=u)
        F_reac = v * expr * dx
        forms.append(F_reac)
    if not is_zero(j):
        F_src = v * D_src(j, trial=u) * dx
        forms.append(F_src)
    return forms


def zero_form(
    v,
    mesh,
    dx,
    shape: tuple[int, ...] = (),       
):
    return v * Constant(mesh, 0.0, shape=shape) * dx


class OptionError(ValueError):
    def __init__(self, opt: int):
        super().__init__(f"Invalid option number '{opt}'")