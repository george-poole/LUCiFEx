from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, dx, TestFunction,
    inner, grad, TestFunction,
    dS, div, jump, lt, conditional, FacetNormal,
)

from dolfinx.fem import FunctionSpace
from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, FiniteDifference, FiniteDifferenceArgwise, 
    DT, AB1,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions



def advection(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
) -> tuple[Form, Form]:
    """
    `∂u/∂t + 𝐚·∇u = 0`
    """
    v = TestFunction(u.function_space)
    Ft = v * DT(u, dt) * dx
    Fa = v * inner(a, grad(D_adv(u))) * dx
    return Fa, Ft


def advection_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    r: Function | Constant | None = None,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = R`
    """
    forms = list(advection(u, dt, a, D_adv))
    if r is not None:
        v = TestFunction(u.function_space)
        r = D_reac(D_reac, r)
        F_reac = -v * r * dx
        forms.append(F_reac)
    return forms


def advection_dg(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_adv: FiniteDifference,
    bcs: BoundaryConditions | None = None,
    adv_dx: int = 0,
    adv_dS: int= 0,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = 0`
    """
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)

    F_dt = v * DT(u, dt) * dx

    match adv_dx:
        case 0:
            F_adv_dx = -inner(grad(v), a) * D_adv(u) * dx
        case 1: 
            F_adv_dx = -div(v * a) * D_adv(u) * dx

    match adv_dS:
        case 0:
            F_adv_dS = jump(v) * jump(0.5 * (inner(a, n) + abs(inner(a, n))) * D_adv(u)) * dS

    forms = [F_dt, F_adv_dx, F_adv_dS]

    if bcs is not None:
        ds, u_inflow = bcs.boundary_data(u.function_space, 'dirichlet')
        inflow = lambda uI: conditional(lt(inner(a, n), 0), uI, 0)
        F_inflow = sum([v * inner(a, n) * inflow(uI) * ds(i) for i, uI in u_inflow])
        ds_outflow = ds(len(u_inflow))
        F_outflow = v * inner(a, n) * D_adv(u) * ds_outflow
        forms.extend([F_inflow, F_outflow])

    return forms


def advection_reaction_dg(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_adv: FiniteDifference,
    r: Function | Constant | None = None,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = R`
    """
    forms = advection_dg(u, dt, a, D_adv, bcs)
    if r is not None:
        v = TestFunction(u.function_space)
        F_reac = -v * D_reac(r) * dx
        forms.append(F_reac)
    return forms

