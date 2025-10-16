from typing import Callable

from ufl.core.expr import Expr
from ufl import dx, Form, inner, TestFunction, div, FacetNormal

from lucifex.fdm import DT, FiniteDifference, apply_finite_difference
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.fdm import (
    FunctionSeries, ConstantSeries,
    FiniteDifference, AB1, Series, 
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import BoundaryConditions

from .supg import supg_diffusivity, supg_velocity, supg_tau, supg_reaction


def advection_diffusion(
    c: FunctionSeries,
    dt: Constant | ConstantSeries,
    u: FunctionSeries,
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
    `∂c/∂t + 𝐮·∇c = ∇·(D·∇c)`
    
    `∂c/∂t + (1/ϕ)𝐮·∇c = (1/ϕ)∇·(D·∇c)`
    """
    if isinstance(d, Series):
        d = D_disp(d)
    if isinstance(phi, Series):
        phi = D_phi(phi)

    v = TestFunction(c.function_space)
    dcdt, adv, diff = advection_diffusion_residuals(
        c, dt, u, d, D_adv, D_diff, phi
    )

    F_dcdt = v * dcdt * dx
    F_adv = v * adv * dx
    F_diff = inner(grad(v / phi), d * grad(D_diff(c))) * dx

    forms = [F_dcdt, F_adv, F_diff]

    if bcs is not None:
        ds, c_neumann = bcs.boundary_data(c.function_space, 'neumann')
        F_neumann = sum([-v * (1/phi) * cN * ds(i) for i, cN in c_neumann])
        forms.append(F_neumann)

    if supg is not None:
        u_eff = (1 / phi) * supg_velocity(u, d, D_adv, D_diff)
        d_eff = (1 / phi) *  supg_diffusivity(d, D_diff)
        tau = supg_tau(supg, c.function_space.mesh, u_eff, d_eff)        
        res = dcdt + adv + diff
        F_res = tau * inner(grad(v), u_eff) * res * dx
        forms.append(F_res)

    return forms


def advection_diffusion_reaction(
    c: FunctionSeries,
    dt: Constant,
    u: FunctionSeries,
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
    `∂c/∂t + 𝐮·∇c = ∇·(D·∇c) + R`

    `∂c/∂t + (1/ϕ)𝐮·∇c = (1/ϕ)∇·(D·∇c) + (1/ϕ)R`
    """
    if isinstance(phi, Series):
        phi = D_phi(phi)
        
    forms = advection_diffusion(c, dt, u, d, D_adv, D_diff, D_disp, D_phi, phi, bcs, supg=None)

    v = TestFunction(c.function_space)
    r = apply_finite_difference(D_reac, r, c)
    reac = -(1 / phi) * r
    F_reac = v * reac * dx

    forms.append(F_reac)

    if supg is not None:
        u_eff = (1 / phi) * supg_velocity(u, d, D_adv, D_diff)
        d_eff = (1 / phi) * supg_diffusivity(d, D_diff)
        r_eff = 0 # FIXME (1 / phi) * supg_reaction(dt, Da, D_reac)
        tau = supg_tau(supg, c.function_space.mesh, u_eff, d_eff, r_eff)   
        dcdt, adv, diff = advection_diffusion_residuals(
            c, dt, u, d, D_adv, D_diff, phi
        )
        res = dcdt + adv + diff + reac
        F_res = tau * inner(grad(v), u_eff) * res * dx
        forms.append(F_res)

    return forms


def advection_diffusion_residuals(
    c: FunctionSeries,
    dt: Constant,
    u: FunctionSeries,
    d: Function | Expr,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    phi: Series | Function | Expr | float = 1,
) -> tuple[Expr, Expr, Expr]:
    
    dcdt = DT(c, dt)

    match D_adv:
        case D_adv_u, D_adv_c:
            adv = (1 / phi) * inner(D_adv_u(u, False), grad(D_adv_c(c)))
        case D_adv:
            adv = (1 / phi) * D_adv(inner(u, grad(c)))

    diff = -(1/phi) * div(d * grad(D_diff(c)))

    return dcdt, adv, diff


def advective_flux(
    c: Function,
    u: Function | Constant,
) -> Expr:
    """
    `fᵁ = (𝐧·𝐮)c`

    for the flux integral

    `Fᵁ = ∫ fᵁ ds` 
    """
    n = FacetNormal(c.function_space.mesh)
    return inner(n, u * c)


def diffusive_flux(
    c: Function,
    d: Function | Constant,
) -> Expr:
    """
    `fᴰ = 𝐧·(D·∇c)`

    for the flux integral

    `Fᴰ = ∫ fᴰ ds`
    """
    n = FacetNormal(c.function_space.mesh)
    return inner(n, d * grad(c))


def flux(
    c: Function,
    u: Function | Constant, 
    d: Function,
) -> tuple[Expr, Expr]:
    """
    Advective flux 
    `fᵁ = (𝐧·𝐮)c`

    and diffusive flux
    `fᴰ = 𝐧·(D·∇c)`

    for the flux integrals

    `Fᵁ = ∫ fᵁ ds`, `Fᴰ = ∫ fᴰ ds`
    """
    return advective_flux(c, u), diffusive_flux(c, d)

