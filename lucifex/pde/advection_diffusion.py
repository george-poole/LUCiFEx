from typing import Callable

import numpy as np
from ufl.core.expr import Expr
from ufl.geometry import CellDiameter
from ufl import (dx, dS, Form, inner, TestFunction, div, 
    Form, CellDiameter, FacetNormal, jump, avg, dot, conditional, gt,
)

from lucifex.fdm import DT, FiniteDifference, FiniteDifferenceArgwise, FiniteDifferenceDerivative
from lucifex.fem import Function, Constant
from lucifex.fdm import (
    DT, AB1, FiniteDifference, FunctionSeries, ConstantSeries, 
    Series, FiniteDifferenceArgwise)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import BoundaryConditions
from lucifex.utils import mesh_integral

from .supg import supg_stabilization


def advection_diffusion(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    a: Series | Function | Expr,
    d: Series | Function | Expr,
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    D_diff: FiniteDifference,
    D_disp: FiniteDifference = AB1,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    supg: str | Callable | None = None,
    h: str = 'hdiam',
) -> list[Form]:
    """    
    `∂u/∂t + (1/ϕ)𝐚·∇u = (1/ϕ)∇·(D·∇u)`
    """
    d = D_disp(d)
    phi = D_phi(phi)

    v = TestFunction(u.function_space)
    dudt, adv, diff = advection_diffusion_residuals(
        u, dt, a, d, DT, D_adv, D_diff, phi
    )

    F_dt = v * dudt * dx
    F_adv = v * adv * dx
    F_diff = inner(grad(v / phi), d * grad(D_diff(u))) * dx

    forms = [F_dt, F_adv, F_diff]

    if bcs is not None:
        ds, u_neumann = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * (1/phi) * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)

    if supg is not None:
        res = dudt + adv + diff
        F_supg = supg_stabilization(supg, v, res, h, a, d, dt=dt, D_adv=D_adv, D_diff=D_diff, phi=phi)
        forms.append(F_supg)

    return forms


def advection_diffusion_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: Series | Function | Expr,
    d: Series | Function | Expr,
    r: Function | Expr | Series | tuple[Callable, tuple],
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    D_diff: FiniteDifference,
    D_reac: FiniteDifference | FiniteDifferenceArgwise,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_disp: FiniteDifference = AB1,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    j: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    bcs: BoundaryConditions | None = None,
    supg: str | Callable | None = None,
    h: str = 'hdiam',
) -> list[Form]:
    """
    `∂u/∂t + (1/ϕ)𝐚·∇u = (1/ϕ)∇·(D·∇u) + (1/ϕ)R + (1/ϕ)J`
    """    
    forms = advection_diffusion(
        u, dt, a, d, D_adv, D_diff, D_disp, D_phi, phi, bcs, supg=None,
    )
    v = TestFunction(u.function_space)
    d = D_disp(d)
    phi = D_phi(phi)

    reac = 0
    if r is not None:
        reaction = lambda r, u: r * u
        if isinstance(D_reac, FiniteDifference):
            reac = -(1 / phi) *  D_reac(reaction(r, u))
        else:
            reac = -(1 / phi) * D_reac(r, u, reaction, trial=u)
        F_reac = v * reac * dx
        forms.append(F_reac)

    src = 0 
    if j is not None:
        j = D_src(j, trial=u)
        src = -(1 / phi) * j
        F_src = v * src * dx
        forms.append(F_src)

    if supg is not None:
        dcdt, adv, diff = advection_diffusion_residuals(
            u, dt, a, d, DT, D_adv, D_diff, phi
        )
        res = dcdt + adv + diff + reac + src
        F_supg = supg_stabilization(supg, v, res, h, a, d, r, dt, D_adv, D_diff, D_reac, DT, phi=phi) 
        forms.append(F_supg)

    return forms


def advection_diffusion_residuals(
    u: FunctionSeries,
    dt: Constant,
    a: FunctionSeries,
    d: Function | Expr,
    D_dt: FiniteDifferenceDerivative,
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    D_diff: FiniteDifference,
    phi: Series | Function | Expr | float = 1,
) -> tuple[Expr, Expr, Expr]:
    
    dudt = D_dt(u, dt)

    advection = lambda a, u: inner(a, grad(u))
    if isinstance(D_adv, FiniteDifference):
        adv = (1 / phi) *  D_adv(advection(a, u))
    else:
        adv = (1 / phi) * D_adv(a, u, advection, trial=u)

    diff = -(1/phi) * div(d * grad(D_diff(u)))

    return dudt, adv, diff


# TODO debug, debug, debug
def advection_diffusion_dg(
    u: FunctionSeries,
    dt: Constant,
    a: FunctionSeries,
    d: Function | Expr | Series, 
    alpha: Constant | float,
    gamma: Constant | float,
    D_adv: FiniteDifferenceArgwise,
    D_diff: FiniteDifference,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    adv_dx: int = 0,
    adv_dS: int= 0,
) -> list[Form]:
    if bcs is None:
        bcs = BoundaryConditions()
    ds, u_dirichlet, u_neumann = bcs.boundary_data(u.function_space, 'dirichlet', 'neumann')

    v = TestFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)
    n = FacetNormal(u.function_space.mesh)
    if isinstance(alpha, float):
        alpha = Constant(u.function_space.mesh, alpha)
    if isinstance(gamma, float):
        gamma = Constant(u.function_space.mesh, gamma)

    F_dudt = v * DT(u, dt) * dx

    D_adv_a, D_adv_u = D_adv
    a = D_adv_a(a)
    phi = D_phi(phi)
    d = D_phi(d)
    outflow = conditional(gt(dot(a, n), 0), 1, 0)

    match adv_dx:
        case 0:
            F_adv_dx = -inner(grad(v / phi), a)  * D_adv_u(u) * dx
        case 1:
            F_adv_dx = -div(v * D_adv_a(a) / phi) * D_adv_u(u) * dx

    match adv_dS:
        case 0:
            F_adv_dS = jump(v) * jump(0.5 * (inner(a, n) + abs(inner(a, n))) * D_adv_u(u)) * dS
        case 1:
            F_adv_dS = 2 * jump(v / phi), avg(outflow * inner(a, n) * D_adv_u(u)) * dS

    F_adv_ds = inner(v / phi, outflow * inner(a * D_adv_u(u), n)) * ds 
    F_adv_ds += sum([inner(v / phi, (1 - outflow) * inner(a * uD, n)) * ds(i) for i, uD in u_dirichlet])
    F_adv = F_adv_dx + F_adv_dS + F_adv_ds

    F_diff_dx = inner(grad(v / phi), d * grad(D_diff(u))) * dx
    F_diff_dS = -inner(jump(v / phi, n), avg(d * grad(D_diff(u)))) * dS
    F_diff_dS += -inner(avg(d * grad(v / phi)), jump(D_diff(u), n)) * dS
    F_diff_dS += (alpha / avg(h)) * inner(jump(v / phi, n), jump(D_diff(u), n)) * dS
    F_diff_ds = sum([-inner(d * grad(v / phi), (D_diff(u) - uD) * n) * ds(i) for i, uD in u_dirichlet])
    F_diff_ds += sum([-inner(v * n / phi, d * grad(D_diff(u))) * ds(i) for i, uD in u_dirichlet])
    F_diff_ds += sum([(gamma / h) * v * (D_diff(u) - uD) * ds(i) for i, uD in u_dirichlet])
    F_diff_ds += sum([-(v / phi) * uN * ds(i) for i, uN in u_neumann])
    F_diff = F_diff_dx + F_diff_dS + F_diff_ds

    return [F_dudt, F_adv, F_diff]


def advection_diffusion_reaction_dg(
    u: FunctionSeries,
    dt: Constant,
    a,
    d,
    r,
    alpha: float,
    gamma: float,
    D_adv: FiniteDifferenceArgwise,
    D_diff: FiniteDifference,
    D_reac: FiniteDifference | FiniteDifferenceArgwise,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    j: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    
    forms = advection_diffusion_dg(u, dt, a, d, alpha, gamma, D_adv, D_diff, D_phi, phi, bcs)

    v = TestFunction(u.function_space)
    phi = D_phi(phi)
    r = D_reac(r, trial=u)
    F_reac = -v  * (r / phi) * dx
    forms.append(F_reac)

    if r is not None:
        reaction = lambda r, u: r * u
        if isinstance(D_reac, FiniteDifference):
            F_reac = -v * (1 / phi) *  D_reac(reaction(r, u)) * dx
        else:
            F_reac = -v * (1 / phi) * D_reac(r, u, reaction, trial=u) * dx
        forms.append(F_reac)

    if j is not None:
        j = D_src(j, trial=u)
        F_src = -v * (1 / phi) * j * dx
        forms.append(F_src)

    return forms


@mesh_integral
def advective_flux(
    u: Function,
    a: Function | Constant,
) -> Expr:
    """
    `Fᵁ = ∫ (𝐧·𝐚)u ds` 
    """
    n = FacetNormal(u.function_space.mesh)
    return inner(n, a * u)


@mesh_integral
def diffusive_flux(
    u: Function,
    d: Function | Constant,
) -> Expr:
    """
    `Fᴰ = ∫ 𝐧·(D·∇u) ds`
    """
    n = FacetNormal(u.function_space.mesh)
    return inner(n, d * grad(u))


@mesh_integral
def flux(
    u: Function,
    a: Function | Constant, 
    d: Function,
) -> tuple[Expr, Expr]:
    """
    `Fᵁ = ∫ (𝐧·𝐚)u ds`, `Fᴰ = ∫ 𝐧·(D·∇u) ds`
    """
    return advective_flux(u, a), diffusive_flux(u, d)