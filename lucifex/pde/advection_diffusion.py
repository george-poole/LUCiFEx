from typing import Callable

from ufl.core.expr import Expr
from ufl.geometry import CellDiameter, GeometricCellQuantity
from ufl import (Measure, Form, inner, TestFunction, TrialFunction, 
    Form, CellDiameter, FacetNormal,
)

from lucifex.fdm import DT, FiniteDifference, FiniteDifferenceArgwise, FiniteDifferenceDerivative
from lucifex.fem import Function, Constant
from lucifex.fdm import (
    DT, AB1, FiniteDifference, FunctionSeries, ConstantSeries, 
    Series, FiniteDifferenceArgwise)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import BoundaryConditions
from lucifex.utils import mesh_integral

from .supg import supg_form
from .cg import derivative_form, advection_form, diffusion_forms, reaction_forms
from .dg import dg_advection_forms, dg_diffusion_forms


def advection_diffusion(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    a: Series | Function | Expr,
    d: Series | Function | Expr,
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    D_diff: FiniteDifference | FiniteDifferenceArgwise,
    D_phi: FiniteDifference = AB1,
    D_dt: FiniteDifferenceDerivative = DT,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    supg: str | Callable | None = None,
    h: str = 'hdiam',
) -> list[Form]:
    """    
    `∂u/∂t + (1/ϕ)𝐚·∇u = (1/ϕ)∇·(D·∇u)`
    """
    return dg_advection_diffusion_reaction(
        u, dt, a, d, 
        D_adv=D_adv, D_diff=D_diff, D_dt=D_dt, D_phi=D_phi,
        phi=phi, bcs=bcs, supg=supg, h=h,
    )


def dg_advection_diffusion_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: Series | Function | Expr,
    d: Series | Function | Expr,
    r: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    j: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_phi: FiniteDifference = AB1,
    D_dt: FiniteDifferenceDerivative = DT,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    supg: str | Callable | None = None,
    h: str = 'hdiam',
) -> list[Form]:
    """
    `∂u/∂t + (1/ϕ)𝐚·∇u = (1/ϕ)∇·(D·∇u) + (1/ϕ)Ru + (1/ϕ)J`
    """
    v = TestFunction(u.function_space)
    dx = Measure('dx', u.function_space.mesh)
    phi = D_phi(phi)
    forms = [
        derivative_form(v, u, dt, D_dt, dx),
        advection_form(v/phi, u, a, D_adv, dx),
        *diffusion_forms(-v/phi, u, d, D_diff, bcs, dx),
        *reaction_forms(-v/phi, u, r, j, D_reac, D_src, dx)
    ]
    if supg is not None:
        terms = [
            derivative_form(1, u, dt, D_dt),
            advection_form(1/phi, u, a, D_adv),
            *diffusion_forms(-1/phi, u, d, D_diff, divergence=True),
            *reaction_forms(-v/phi, u, r, j, D_reac, D_src),
        ]
        res = sum(terms)
        F_supg = supg_form(
            supg, v, res, h, a, d, r, dt, D_adv, D_diff, D_reac, D_dt, phi, dx=dx)
        forms.append(F_supg)

    return forms


def steady_advection_diffusion(
    u: Function | FunctionSeries, 
    a: Function | Constant, 
    d: Constant, 
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    supg: str | Callable | None = None,
    petrov: str | Callable | None = None,
    h: GeometricCellQuantity | str = 'hdiam',
) -> list[Form]:
    """
    `𝐚·∇u = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    dx = Measure('dx', u.function_space.mesh)
    forms = [
        advection_form(v, u_trial, a, dx=dx),
        *diffusion_forms(-v, u_trial, d, bcs=bcs, dx=dx),
        *reaction_forms(-v, u_trial, r, j, dx=dx)
    ]
    if supg is not None:
        terms = [
            advection_form(1, u_trial, a),
            *diffusion_forms(-1, u_trial, d, divergence=True),
            *reaction_forms(-1, u_trial, r, j),
        ]
        res = sum(terms)
        F_supg = supg_form(supg, v, res, h, a, d, r, petrov_func=petrov, dx=dx)
        forms.append(F_supg)

    return forms


# TODO debug, debug, debug
def dg_advection_diffusion(
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
    return dg_advection_diffusion_reaction(
        u, dt, a, d, alpha=alpha, gamma=gamma, 
        D_adv=D_adv, D_diff=D_diff, D_phi=D_phi,
        phi=phi, bcs=bcs,
        adv_dx=adv_dx, adv_dS=adv_dS,
    )


def dg_advection_diffusion_reaction(
    u: FunctionSeries,
    dt: Constant,
    a,
    d,
    r,
    j,
    alpha: float,
    gamma: float,
    D_adv: FiniteDifferenceArgwise,
    D_diff: FiniteDifference,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    adv_dx: int = 0,
    adv_dS: int= 0,
) -> list[Form]:
    
    phi = D_phi(phi)
    v = TestFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)
    n = FacetNormal(u.function_space.mesh)
    dx = Measure('dx', u.function_space.mesh)
    dS = Measure('dS', u.function_space.mesh)

    return [
        derivative_form(),
        *dg_advection_forms(),
        *dg_diffusion_forms(),
        *reaction_forms(),
    ]


def dg_steady_advection_diffusion(
    u: Function, 
    a: Function | Constant, 
    d: Function | Constant, 
    alpha: float | Constant,
    gamma: float | Constant,
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    adv_dx: int = 0,
    adv_dS: int= 0,
) -> list[Form]:
    """
    `𝐚·∇u = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)
    n = FacetNormal(u.function_space.mesh)
    ds, u_dirichlet, u_neumann = bcs.boundary_data(u.function_space, 'dirichlet', 'neumann')

    return [
        *dg_advection_forms(),
        *dg_diffusion_forms(),
        *reaction_forms(),
    ]


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
    `Fᴰ = ∫ 𝐧·(-D·∇u) ds`
    """
    n = FacetNormal(u.function_space.mesh)
    return inner(n, -d * grad(u))


@mesh_integral
def flux(
    u: Function,
    a: Function | Constant, 
    d: Function,
) -> tuple[Expr, Expr]:
    """
    `Fᵁ = ∫ (𝐧·𝐚)u ds`, `Fᴰ = ∫ 𝐧·(-D·∇u) ds`
    """
    return advective_flux(u, a), diffusive_flux(u, d)