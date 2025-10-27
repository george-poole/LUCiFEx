from typing import Callable

from ufl.core.expr import Expr
from ufl import dx, Form, inner, TestFunction, div, FacetNormal

from lucifex.fdm import DT, FiniteDifference, apply_finite_difference
from lucifex.fem import SpatialFunction as Function, SpatialConstant as Constant
from lucifex.fdm import (
    FunctionSeries, ConstantSeries,
    FiniteDifference, AB1, Series, 
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import BoundaryConditions
from lucifex.utils import integral

from .supg import supg_diffusivity, supg_velocity, supg_tau, supg_reaction


def advection_diffusion(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    a: FunctionSeries,
    d: Series | Function | Expr,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    D_disp: FiniteDifference = AB1,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    supg: str | None = None,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = ∇·(D·∇u)`
    
    `∂u/∂t + (1/ϕ)𝐚·∇u = (1/ϕ)∇·(D·∇u)`
    """
    if isinstance(d, Series):
        d = D_disp(d)
    if isinstance(phi, Series):
        phi = D_phi(phi)

    v = TestFunction(u.function_space)
    dudt, adv, diff = advection_diffusion_residuals(
        u, dt, a, d, D_adv, D_diff, phi
    )

    F_dt = v * dudt * dx
    F_adv = v * adv * dx
    F_diff = inner(grad(v / phi), d * grad(D_diff(u))) * dx

    forms = [F_dt, F_adv, F_diff]

    if bcs is not None:
        ds, c_neumann = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * (1/phi) * cN * ds(i) for i, cN in c_neumann])
        forms.append(F_neumann)

    if supg is not None:
        u_eff = (1 / phi) * supg_velocity(a, d, D_adv, D_diff)
        d_eff = (1 / phi) *  supg_diffusivity(d, D_diff)
        tau = supg_tau(supg, u.function_space.mesh, u_eff, d_eff)        
        res = dudt + adv + diff
        F_res = tau * inner(grad(v), u_eff) * res * dx
        forms.append(F_res)

    return forms


def advection_diffusion_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: FunctionSeries,
    d: Series | Function | Expr,
    r: Function | Expr | Series | tuple[Callable, tuple],
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    D_reac: FiniteDifference,
    D_disp: FiniteDifference = AB1,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    supg: str | None = None,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = ∇·(D·∇u) + R`

    `∂u/∂t + (1/ϕ)𝐚·∇u = (1/ϕ)∇·(D·∇u) + (1/ϕ)R`
    """
    if isinstance(phi, Series):
        phi = D_phi(phi)
        
    forms = advection_diffusion(u, dt, a, d, D_adv, D_diff, D_disp, D_phi, phi, bcs, supg=None)

    v = TestFunction(u.function_space)
    r = apply_finite_difference(D_reac, r, u)
    reac = -(1 / phi) * r
    F_reac = v * reac * dx

    forms.append(F_reac)

    if supg is not None:
        u_eff = (1 / phi) * supg_velocity(a, d, D_adv, D_diff)
        d_eff = (1 / phi) * supg_diffusivity(d, D_diff)
        r_eff = 0 # FIXME (1 / phi) * supg_reaction(dt, Da, D_reac)
        tau = supg_tau(supg, u.function_space.mesh, u_eff, d_eff, r_eff)   
        dcdt, adv, diff = advection_diffusion_residuals(
            u, dt, a, d, D_adv, D_diff, phi
        )
        res = dcdt + adv + diff + reac
        F_res = tau * inner(grad(v), u_eff) * res * dx
        forms.append(F_res)

    return forms


def advection_diffusion_residuals(
    u: FunctionSeries,
    dt: Constant,
    a: FunctionSeries,
    d: Function | Expr,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    phi: Series | Function | Expr | float = 1,
) -> tuple[Expr, Expr, Expr]:
    
    dudt = DT(u, dt)

    match D_adv:
        case D_adv_u, D_adv_c:
            adv = (1 / phi) * inner(D_adv_u(a, False), grad(D_adv_c(u)))
        case D_adv:
            adv = (1 / phi) * D_adv(inner(a, grad(u)))

    diff = -(1/phi) * div(d * grad(D_diff(u)))

    return dudt, adv, diff


@integral
def advective_flux(
    u: Function,
    a: Function | Constant,
) -> Expr:
    """
    `Fᵁ = ∫ (𝐧·𝐚)u ds` 
    """
    n = FacetNormal(u.function_space.mesh)
    return inner(n, a * u)


@integral
def diffusive_flux(
    u: Function,
    d: Function | Constant,
) -> Expr:
    """
    `Fᴰ = ∫ 𝐧·(D·∇u) ds`
    """
    n = FacetNormal(u.function_space.mesh)
    return inner(n, d * grad(u))


@integral
def flux(
    u: Function,
    a: Function | Constant, 
    d: Function,
) -> tuple[Expr, Expr]:
    """
    `Fᵁ = ∫ (𝐧·𝐚)u ds`, `Fᴰ = ∫ 𝐧·(D·∇u) ds`
    """
    return advective_flux(u, a), diffusive_flux(u, d)
