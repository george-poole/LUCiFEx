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
from lucifex.fem import SpatialFunction as Function, SpatialConstant as Constant
from lucifex.solver import (
    BoundaryConditions, BVP, IBVP,
    bvp, ibvp,
)

from .classic import poisson
from .transport import advection_diffusion_reaction
from .constitutive import strain


def ipcs_1(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: ConstantSeries | Constant,
    deviatoric_stress: Callable[[Function], Expr],
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    sigma_bcs: BoundaryConditions | None = None,
    adv_coeff:  Constant | float = 1,
    p_coeff: Constant | float = 1,
) -> list[Form]:
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)
    dim = u.shape[0]
    epsilon = strain(v)

    F_dudt = inner(v, DT(u, dt)) * dx
    F_adv = adv_coeff * inner(v, D_adv(dot(u, nabla_grad(u)))) * dx
    tau = deviatoric_stress(D_visc(u))
    sigma = -p_coeff * p[0] * Identity(dim) + tau

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
    deviatoric_stress: Callable[[Function], Expr],
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
    `∂𝐮/∂t + 𝐮·∇𝐮 = -∇p + ∇·𝜏(𝐮) + 𝐟`

    If `adv_coeff` argument specified, then `𝐮·∇𝐮 -> A 𝐮·∇𝐮` with constant `A`. \\
    If `p_coeff` argument specified, then `∇p -> P ∇p` with constant `P`.
    """
    ipcs1_solver = ibvp(ipcs_1, bcs=u_bcs)(
        u, p, dt, deviatoric_stress, D_adv, D_visc, D_force, f, sigma_bcs, adv_coeff, p_coeff,
    )
    ipcs2_solver = bvp(ipcs_2, bcs=p_bcs, future=True)(
        p, u, dt, p_coeff,
    )
    ipcs3_solver = bvp(ipcs_3, future=True, overwrite=True)(
        u, p, dt, p_coeff,
    )
    return ipcs1_solver, ipcs2_solver, ipcs3_solver


def chorin_1(
    u: FunctionSeries,
    dt: ConstantSeries | Constant,
    deviatoric_stress: Callable[[Function], Expr],
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    adv_coeff: Constant | float = 1,
) -> list[Form]:
    v = TestFunction(u.function_space)
    F_dudt = inner(v, DT(u, dt)) * dx
    F_adv = adv_coeff * inner(v, D_adv(dot(u, nabla_grad(u)))) * dx
    tau = deviatoric_stress(D_visc(u))
    F_visc = inner(grad(v), tau) * dx
    
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
    deviatoric_stress: Callable[[Function], Expr],
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
    u_bcs: BoundaryConditions | None = None,
    p_bcs: BoundaryConditions | None = None,    
    adv_coeff: Constant | float = 1,
    p_coeff: Constant | float = 1,
):
    """
    `∂𝐮/∂t + 𝐮·∇𝐮 = -∇p + ∇·𝜏(𝐮) + 𝐟` \\
    `∇·𝐮 = 0`

    If `adv_coeff` argument specified, then `𝐮·∇𝐮 -> A 𝐮·∇𝐮` with constant `A`. \\
    If `visc_coeff` argument specified, then `∇²𝐮 -> V ∇²𝐮` with constant `V`. \\
    If `p_coeff` argument specified, then `∇p -> P ∇p` with constant `P`.
    """
    chorin1_solver = ibvp(chorin_1, bcs=u_bcs)(
        u, dt, deviatoric_stress, D_adv, D_visc, D_force, f, adv_coeff,
    )
    chorin2_solver = bvp(chorin_2, bcs=p_bcs, future=True)(
        p, u, dt, p_coeff,
    )
    chorin3_solver = bvp(chorin_3, future=True, overwrite=True)(
        u, p, dt, p_coeff,
    )
    return chorin1_solver, chorin2_solver, chorin3_solver


def vorticity_poisson(
    psi: Function,
    omega: Function,
) -> tuple[Form, Form]:
    return poisson(psi, omega)


def vorticity_transport(
    omega: FunctionSeries,
    dt: Constant,
    u: FunctionSeries,
    rho: Constant,
    mu: Constant,
    D_adv: FiniteDifference,
    D_diff: FiniteDifference,
    D_reac: FiniteDifference = FE,
    fx: Function | None = None,
    fy: Function | None = None,
) -> list[Form]:
    _none = (None, 0)
    d = mu / rho
    r = 0
    if not fx in _none:
        r -= Dx(fx, 1) / rho
    if not fy is not _none:
        r += Dx(fy, 0) / rho
    return advection_diffusion_reaction(omega, dt, u, d, r, D_adv, D_diff, D_reac)