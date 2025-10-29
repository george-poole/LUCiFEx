from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, dx, TestFunction, TrialFunction,
    inner, grad, TestFunction, TrialFunction,
    dS, div, jump, lt, conditional, FacetNormal
)

from dolfinx.fem import FunctionSpace
from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, FiniteDifference, FiniteDifferenceTuple, 
    DT, AB1,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions


def poisson(
    u: Function | FunctionSpace,
    f: Function | Constant | Expr,
) -> tuple[Form, Form]:
    """
    `∇²u = f`
    """
    if isinstance(u, FunctionSpace):
        fs = u
    else:
        fs = u.function_space
    v = TestFunction(fs)
    u_trial = TrialFunction(fs)
    F_lhs = -inner(grad(v), grad(u_trial)) * dx
    F_rhs = -inner(v, f) * dx
    return F_lhs, F_rhs


def diffusion(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_diff: FiniteDifference, # TODO tuple
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t = ∇·(D·∇u)`

    `(uⁿ⁺¹ - uⁿ) / Δtⁿ = ∇·(𝒟(D·∇u))`
    """
    v = TestFunction(u.function_space)
    Ft = v * DT(u, dt) * dx
    Fd = inner(grad(v), d * grad(D_diff(u))) * dx
    forms = [Ft, Fd]
    if bcs is not None:
        ds, u_neumann = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)
    return forms


def diffusion_reaction(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_diff: FiniteDifference,
    r: Function | Constant | None = None,
    D_reac: FiniteDifference | FiniteDifferenceTuple = AB1,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t = ∇·(D·∇u) + R`

    `(uⁿ⁺¹ - uⁿ) / Δtⁿ = ∇·(𝒟(D·∇u)) + 𝒟(R)`
    """
    forms = diffusion(u, dt, d, D_diff, bcs)
    if r is not None:
        v = TestFunction(u.function_space)
        F_reac = -v * D_reac(r) * dx
        forms.append(F_reac)
    return forms


def advection(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_adv: FiniteDifference | FiniteDifferenceTuple,
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
    D_adv: FiniteDifference | FiniteDifferenceTuple,
    r: Function | Constant | None = None,
    D_reac: FiniteDifference | FiniteDifferenceTuple = AB1,
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
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = 0`
    """
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)

    F_dt = v * DT(u, dt) * dx

    aOut = 0.5 * (inner(a, n) + abs(inner(a, n)))
    Fa_dx = - D_adv(u) * div(v * a) * dx
    Fa_dS = jump(v) * jump(aOut * D_adv(u)) * dS

    forms = [F_dt, Fa_dx, Fa_dS]

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
    D_fdm: FiniteDifference,
    r: Function | Constant | None = None,
    D_reac: FiniteDifference | FiniteDifferenceTuple = AB1,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = R`
    """
    forms = advection_dg(u, dt, a, D_fdm, bcs)
    if r is not None:
        v = TestFunction(u.function_space)
        F_reac = -v * D_reac(r) * dx
        forms.append(F_reac)
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