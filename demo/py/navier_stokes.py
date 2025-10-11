from typing import Callable

from ufl import (
    dx, ds, Form, FacetNormal,
    TrialFunction, TestFunction, Identity, sym,
)
from ufl.core.expr import Expr

from lucifex.fdm import (
    DT, FE, CN, FiniteDifference, 
    FunctionSeries, ConstantSeries, Series,
)
from lucifex.fdm.ufl_operators import inner, div, nabla_grad, dot, grad
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.solver import (
    BoundaryConditions, BVP, IBVP,
    bvp_solver, ibvp_solver,
)

def strain(u: Function | Expr) -> Expr:
    """
    `ε(𝐮) = (∇𝐮 + (∇𝐮)ᵀ) / 2` (dimensional)
    """
    return sym(nabla_grad(u))


def newtonian_stress(
    u: Function | Expr, 
    p: Function | Expr,
    mu: Constant,
) -> Expr:
    """
    `σ(𝐮, p) = - pI + 2με(𝐮)` (dimensional)
    """
    dim = u.ufl_shape[0]
    return -p * Identity(dim) + 2 * mu * strain(u)


def ipcs_1(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: ConstantSeries | Constant,
    rho: Constant,
    mu: Constant,
    stress: Callable[[Function, Function, Constant], Expr],
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    sigma_bcs: BoundaryConditions | None = None,
) -> list[Form]:
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)
    epsilon = strain(v)
    sigma = stress(D_visc(u), p[0], mu)

    F_dudt = rho * inner(v, DT(u, dt)) * dx
    F_adv = rho * inner(v, D_adv(dot(u, nabla_grad(u)))) * dx
    F_visc = inner(epsilon, sigma) * dx 
    if sigma_bcs is None:
        F_ds = -inner(v, dot(n, sigma)) * ds
    else:
        dsN, natural = sigma_bcs.boundary_data(u.function_space, 'natural')
        F_ds = sum([-inner(v, sigmaN) * dsN(i) for i, sigmaN in natural])
    forms = [F_dudt, F_adv, F_visc, F_ds]

    if f is not None:
        if isinstance(f, Series):
            f = D_force(f)
        F_f = -inner(v, f) * dx
        forms.append(F_f)

    return forms


def ipcs_2(
    p: FunctionSeries,
    u: FunctionSeries,
    dt: Constant,
    rho: Constant,
) -> tuple[Form, Form, Form]:
    p_trial = TrialFunction(p.function_space)
    q = TestFunction(p.function_space)
    F_trial = inner(grad(q), grad(p_trial)) * dx
    F_grad = -inner(grad(q), grad(p[0])) * dx
    F_div = q * rho * (1 / dt) * div(u[1])  * dx
    return F_trial, F_grad, F_div
    

def ipcs_3(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    rho: Constant,
) -> tuple[Form, Form]:
    u_trial = TrialFunction(u.function_space)
    v = TestFunction(u.function_space)
    F_dudt = rho * (1 / dt) * inner(v, (u_trial - u[1])) * dx
    F_grad = inner(v, grad(p[1]) - grad(p[0])) * dx
    return F_dudt, F_grad


def ipcs_solvers(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    rho: Constant,
    mu: Constant,
    stress: Callable[[Function, Function, Constant], Expr],
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    u_bcs: BoundaryConditions | None = None,
    p_bcs: BoundaryConditions | None = None,
    sigma_bcs: BoundaryConditions | None = None,
) -> tuple[IBVP, BVP, BVP]:
    """
    `ρ(∂𝐮/∂t + 𝐮·∇𝐮) = ∇·σ(𝐮, p) + 𝐟` \\
    `∇·𝐮 = 0` (dimensional)
    """
    ipcs1_solver = ibvp_solver(ipcs_1, bcs=u_bcs)(
        u, p, dt, rho, mu, stress, D_adv, D_visc, D_force, f, sigma_bcs
    )
    ipcs2_solver = bvp_solver(ipcs_2, bcs=p_bcs, future=True)(
        p, u, dt, rho,
    )
    ipcs3_solver = bvp_solver(ipcs_3, future=True, overwrite=True)(
        u, p, dt, rho,
    )
    return ipcs1_solver, ipcs2_solver, ipcs3_solver


def chorin_1(
    u: FunctionSeries,
    dt: ConstantSeries | Constant,
    rho: Constant,
    mu: Constant,
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
) -> list[Form]:
    v = TestFunction(u.function_space)
    F_dudt = rho * inner(v, DT(u, dt)) * dx
    F_adv = rho * inner(v, D_adv(dot(u, nabla_grad(u)))) * dx
    F_visc = mu * inner(grad(v), grad(D_visc(u))) * dx
    
    forms = [F_dudt, F_adv, F_visc]

    if f is not None:
        if isinstance(f, Series):
            f = D_force(f)
        F_f = -inner(v, f) * dx
        forms.append(F_f)

    return forms


def chorin_2(
    p: FunctionSeries,
    u: FunctionSeries,
    dt: Constant,
    rho: Constant,
) -> tuple[Form, Form]:
    p_trial = TrialFunction(p.function_space)
    q = TestFunction(p.function_space)
    F_trial = inner(grad(q), grad(p_trial)) * dx
    F_div = q * rho * (1 / dt) * div(u[1])  * dx
    return F_trial, F_div


def chorin_3(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    rho: Constant,
) -> tuple[Form, Form]:
    u_trial = TrialFunction(u.function_space)
    v = TestFunction(u.function_space)
    F_dudt = rho * (1 / dt) * inner(v, (u_trial - u[1])) * dx
    F_grad = inner(v, grad(p[1])) * dx
    return F_dudt, F_grad


def chorin_solvers(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    rho: Constant,
    mu: Constant,
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    u_bcs: BoundaryConditions | None = None,
    p_bcs: BoundaryConditions | None = None,    
):
    """
    `ρ(∂𝐮/∂t + 𝐮·∇𝐮) = -∇p + μ∇²𝐮 + 𝐟` \\
    `∇·𝐮 = 0` (dimensional)
    """
    chorin1_solver = ibvp_solver(chorin_1, bcs=u_bcs)(
        u, dt, rho, mu, D_adv, D_visc, D_force, f,
    )
    chorin2_solver = bvp_solver(chorin_2, bcs=p_bcs, future=True)(
        p, u, dt, rho,
    )
    chorin3_solver = bvp_solver(chorin_3, future=True, overwrite=True)(
        u, p, dt, rho,
    )
    return chorin1_solver, chorin2_solver, chorin3_solver


def advection_diffusion(
    c: FunctionSeries,
    dt: Constant,
    u: FunctionSeries,
    Pe: Constant,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂c/∂t + 𝐮·∇c = 1/Pe ∇²c` (dimensionless)
    """
    v = TestFunction(c.function_space)

    F_dcdt = v * DT(c, dt) * dx

    match D_adv:
        case D_adv_u, D_adv_c:
            adv = inner(D_adv_u(u, False), grad(D_adv_c(c)))
        case D_adv:
            adv = D_adv(inner(u, grad(c)))
    F_adv = v * adv * dx

    F_diff = (1/Pe) * inner(grad(v), grad(D_diff(c))) * dx

    forms = [F_dcdt, F_adv, F_diff]

    if bcs is not None:
        ds, c_neumann = bcs.boundary_data(c.function_space, 'neumann')
        F_neumann = sum([-(1 / Pe) * v * cN * ds(i) for i, cN in c_neumann])
        forms.append(F_neumann)

    return forms