from typing import Callable

from ufl import (
    dx, ds, Dx, Form, FacetNormal,
    TrialFunction, TestFunction, Identity, sym,
)
from ufl.core.expr import Expr

from lucifex.fdm import (
    DT, FE, FiniteDifference, 
    FunctionSeries, ConstantSeries, Series,
)
from lucifex.fdm.ufl_operators import inner, div, nabla_grad, dot, grad
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.solver import (
    BoundaryConditions, BVP, IBVP,
    bvp_solver, ibvp_solver,
)

from .constitutive import strain, newtonian_stress


def ipcs_1(
    u: FunctionSeries,
    dt: ConstantSeries | Constant,
    sigma: Function | Expr,
    D_adv: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    sigma_bcs: BoundaryConditions | None = None,
    adv_coeff:  Constant | float = 1,
) -> list[Form]:
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)
    epsilon = strain(v)

    F_dudt = inner(v, DT(u, dt)) * dx
    F_adv = adv_coeff * inner(v, D_adv(dot(u, nabla_grad(u)))) * dx
    F_stress = inner(epsilon, sigma) * dx 
    if sigma_bcs is None:
        F_ds = -inner(v, dot(n, sigma)) * ds
    else:
        dsN, natural = sigma_bcs.boundary_data(u.function_space, 'natural')
        F_ds = sum([-inner(v, sigmaN) * dsN(i) for i, sigmaN in natural])
    forms = [F_dudt, F_adv, F_stress, F_ds]

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
    p_coeff:  Constant | float = 1,
) -> tuple[Form, Form, Form]:
    p_trial = TrialFunction(p.function_space)
    q = TestFunction(p.function_space)
    F_trial = p_coeff * inner(grad(q), grad(p_trial)) * dx
    F_grad = -p_coeff * inner(grad(q), grad(p[0])) * dx
    F_div = q * (1 / dt) * div(u[1])  * dx
    return F_trial, F_grad, F_div
    

def ipcs_3(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    p_coeff:  Constant | float = 1,
) -> tuple[Form, Form]:
    u_trial = TrialFunction(u.function_space)
    v = TestFunction(u.function_space)
    F_dudt = (1 / dt) * inner(v, (u_trial - u[1])) * dx
    F_grad = p_coeff * inner(v, grad(p[1]) - grad(p[0])) * dx
    return F_dudt, F_grad


def ipcs_solvers(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    stress: Callable[[Function, Function], Expr],
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    u_bcs: BoundaryConditions | None = None,
    p_bcs: BoundaryConditions | None = None,
    sigma_bcs: BoundaryConditions | None = None,
    adv_coeff: Constant | float = 1,
    p_coeff: Constant | float = 1,
) -> tuple[IBVP, BVP, BVP]:
    """
    `∇·𝐮 = 0` \\
    `∂𝐮/∂t + 𝐮·∇𝐮 = ∇·σ + 𝐟`

    If `adv_coeff` argument specified, then `𝐮·∇𝐮 -> A 𝐮·∇𝐮` with constant `A`. \\
    If `p_coeff` argument specified, then `∇p -> P ∇p` with constant `P`.
    """
    sigma = stress(D_visc(u), p[0])
    ipcs1_solver = ibvp_solver(ipcs_1, bcs=u_bcs)(
        u, dt, sigma, D_adv, D_force, f, sigma_bcs, adv_coeff,
    )
    ipcs2_solver = bvp_solver(ipcs_2, bcs=p_bcs, future=True)(
        p, u, dt, p_coeff,
    )
    ipcs3_solver = bvp_solver(ipcs_3, future=True, overwrite=True)(
        u, p, dt, p_coeff,
    )
    return ipcs1_solver, ipcs2_solver, ipcs3_solver


def chorin_1(
    u: FunctionSeries,
    dt: ConstantSeries | Constant,
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    adv_coeff: Constant | float = 1,
    visc_coeff: Constant | float = 1,
) -> list[Form]:
    v = TestFunction(u.function_space)
    F_dudt = inner(v, DT(u, dt)) * dx
    F_adv = adv_coeff * inner(v, D_adv(dot(u, nabla_grad(u)))) * dx
    F_visc = visc_coeff * inner(grad(v), grad(D_visc(u))) * dx
    
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
    p_coeff:  Constant | float = 1,
) -> tuple[Form, Form]:
    p_trial = TrialFunction(p.function_space)
    q = TestFunction(p.function_space)
    F_trial = p_coeff * inner(grad(q), grad(p_trial)) * dx
    F_div = q * (1 / dt) * div(u[1])  * dx
    return F_trial, F_div


def chorin_3(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    p_coeff:  Constant | float = 1,
) -> tuple[Form, Form]:
    u_trial = TrialFunction(u.function_space)
    v = TestFunction(u.function_space)
    F_dudt = (1 / dt) * inner(v, (u_trial - u[1])) * dx
    F_grad = p_coeff * inner(v, grad(p[1])) * dx
    return F_dudt, F_grad


def chorin_solvers(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    u_bcs: BoundaryConditions | None = None,
    p_bcs: BoundaryConditions | None = None,    
    adv_coeff: Constant | float = 1,
    visc_coeff: Constant | float = 1,
    p_coeff: Constant | float = 1,
):
    """
    `∂𝐮/∂t + 𝐮·∇𝐮 = -∇p + ∇²𝐮 + 𝐟` \\
    `∇·𝐮 = 0` (dimensional)

    If `adv_coeff` argument specified, then `𝐮·∇𝐮 -> A 𝐮·∇𝐮` with constant `A`. \\
    If `visc_coeff` argument specified, then `∇²𝐮 -> V ∇²𝐮` with constant `V`. \\
    If `p_coeff` argument specified, then `∇p -> P ∇p` with constant `P`.
    """
    chorin1_solver = ibvp_solver(chorin_1, bcs=u_bcs)(
        u, dt, D_adv, D_visc, D_force, f, adv_coeff, visc_coeff,
    )
    chorin2_solver = bvp_solver(chorin_2, bcs=p_bcs, future=True)(
        p, u, dt, p_coeff,
    )
    chorin3_solver = bvp_solver(chorin_3, future=True, overwrite=True)(
        u, p, dt, p_coeff,
    )
    return chorin1_solver, chorin2_solver, chorin3_solver


def streamfunction_vorticity_poisson(
    psi: Function,
    omega: Function,
) -> tuple[Form, Form]:
    v = TestFunction(psi.function_space)
    psi_trial = TrialFunction(psi.function_space)
    F_lhs = -inner(grad(v), grad(psi_trial)) * dx
    F_rhs = -v * omega * dx
    return F_lhs, F_rhs


def vorticity_advection_diffusion(
    omega: FunctionSeries,
    dt: Constant,
    u: FunctionSeries,
    rho: Constant,
    mu: Constant,
    D_adv: FiniteDifference,
    D_diff: FiniteDifference,
    fx: Function | None = None,
    fy: Function | None = None,
) ->list[Form]:
    v = TestFunction(omega.function_space)
    Ft = v * rho * DT(omega, dt) * dx
    Fa = v * rho * D_adv(inner(u, grad(omega))) * dx
    Fd = inner(grad(v), mu * grad(D_diff(omega))) * dx
    forms = [Ft, Fa, Fd]
    if fx is not None:
        F_fx = v * Dx(fx, 1) * dx
        forms.append(F_fx)
    if fy is not None:
        F_fy = -v * Dx(fy, 0) * dx
        forms.append(F_fy)
    return forms