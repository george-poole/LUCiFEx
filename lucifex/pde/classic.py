from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, dx, TestFunction, TrialFunction,
    inner, grad, TestFunction, TrialFunction,
    dS, div, jump, lt, conditional, FacetNormal
)

from dolfinx.fem import FunctionSpace
from lucifex.fdm import (
    DT, FiniteDifference, FunctionSeries,
)
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.solver import BoundaryConditions


def poisson(
    u: Function,
    f: Function | Constant | Expr,
) -> tuple[Form, Form]:
    """
    `∇²u = f`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    F_lhs = -inner(grad(v), grad(u_trial)) * dx
    F_rhs = -inner(v, f) * dx
    return F_lhs, F_rhs


def diffusion(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_fdm: FiniteDifference,
    r: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t = ∇·(D·∇u) + R`
    """
    v = TestFunction(u.function_space)
    Ft = v * DT(u, dt) * dx
    Fd = inner(grad(v), d * grad(D_fdm(u))) * dx
    forms = [Ft, Fd]
    if r is not None:
        F_reac = -v * r * dx
        forms.append(F_reac)
    if bcs is not None:
        ds, u_neumann = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)
    return forms


def advection(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_fdm: FiniteDifference,
    r: Function | Constant | None = None,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = R`
    """
    v = TestFunction(u.function_space)
    Ft = v * DT(u, dt) * dx
    Fa = v * inner(a, grad(D_fdm(u))) * dx # TODO tuple option
    forms = [Fa, Ft]
    if r is not None:
        F_reac = -v * r * dx
        forms.append(F_reac)
    return forms


def advection_dg(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_fdm: FiniteDifference,
    r: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = R`
    """
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)

    F_dt = v * DT(u, dt) * dx

    aOut = 0.5 * (inner(a, n) + abs(inner(a, n)))
    Fa_dx = - D_fdm(u) * div(v * a) * dx
    Fa_dS = jump(v) * jump(aOut * D_fdm(u)) * dS

    forms = [F_dt, Fa_dx, Fa_dS]

    if r is not None:
        F_reac = -v * r * dx
        forms.append(F_reac)
    if bcs is not None:
        ds, u_inflow = bcs.boundary_data(u.function_space, 'dirichlet')
        inflow = lambda uI: conditional(lt(inner(a, n), 0), uI, 0)
        F_inflow = sum([v * inner(a, n) * inflow(uI) * ds(i) for i, uI in u_inflow])
        ds_outflow = ds(len(u_inflow))
        F_outflow = v * inner(a, n) * D_fdm(u) * ds_outflow
        forms.extend([F_inflow, F_outflow])

    return forms


def helmholtz(
    u: Function | FunctionSpace,
    k: Constant | float | None = None,
    f: Function | Constant | Expr | None = None
) -> tuple[Form, Form] | tuple[Form, Form, Form]:
    """
    `∇²u + k²u = f`

    If `k=None` and `f=None`, returns forms for the left and right hand sides
    of the eigenvalue problem `∇²u = λu` where `λ = -k²`.

    Otherwise returns forms for a boundary value problem.
    """
    if isinstance(u, FunctionSpace):
        fs = u
    else:
        fs = u.function_space
    v = TestFunction(fs)
    u_trial = TrialFunction(fs)
    F_lhs = -inner(grad(v), grad(u_trial)) * dx
    F_rhs = v * u * dx
    if k is None and f is None:
        return F_lhs, F_rhs
    else:
        F_rhs = -k * F_rhs
        F_src =  -v * f * dx
        return F_lhs, F_rhs, F_src