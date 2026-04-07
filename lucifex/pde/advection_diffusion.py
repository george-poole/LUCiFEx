from typing import Callable

from ufl.core.expr import Expr
from ufl.geometry import CellDiameter, GeometricCellQuantity
from ufl import (Measure, Form, inner, TestFunction, TrialFunction, 
    Form, CellDiameter, FacetNormal,
)

from lucifex.fdm import DT, FiniteDifference, FiniteDifferenceArgwise, FiniteDifferenceDerivative
from lucifex.fem import Function, Constant
from lucifex.fdm import (
    DT, FE, BE, FiniteDifference, FunctionSeries, ConstantSeries, 
    Series, FiniteDifferenceArgwise)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import mesh_integral

from .stabilization import stabilization_form
from .cg_transport import derivative_form, advection_forms, diffusion_forms, reaction_forms
from .dg_transport import dg_advection_forms, dg_diffusion_forms


def advection_diffusion(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    a: Series | Function | Expr,
    d: Series | Function | Expr,
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    D_diff: FiniteDifference | FiniteDifferenceArgwise,
    D_phi: FiniteDifference = FE,
    D_dt: FiniteDifferenceDerivative = DT,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    tau: str | Callable | None = None,
    Pv: str | Callable | None = None,
    h: str = 'hdiam',
) -> list[Form]:
    """    
    `∂u/∂t + (1/ϕ)𝐚·∇u = (1/ϕ)∇·(D·∇u)`
    """
    return advection_diffusion_reaction(
        u, dt, a, d, 
        D_adv=D_adv, D_diff=D_diff, D_dt=D_dt, D_phi=D_phi,
        phi=phi, bcs=bcs, tau=tau, Pv=Pv, h=h,
    )


def advection_diffusion_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: Series | Function | Expr,
    d: Series | Function | Expr,
    r: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    j: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = FE,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = FE,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = FE,
    D_src: FiniteDifference | FiniteDifferenceArgwise = FE,
    D_phi: FiniteDifference = FE,
    D_dt: FiniteDifferenceDerivative = DT,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    tau: str | Callable | None = None,
    Pv: str | Callable | None = None,
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
        *advection_forms(v/phi, u, a, D_adv, bcs, dx, by_parts=False),
        *diffusion_forms(-v/phi, u, d, D_diff, bcs, dx, by_parts=True),
        *reaction_forms(-v/phi, u, r, j, D_reac, D_src, dx),
    ]
    if tau is not None:
        terms = [
            derivative_form(1, u, dt, D_dt),
            *advection_forms(1/phi, u, a, D_adv, by_parts=False),
            *diffusion_forms(-1/phi, u, d, D_diff, by_parts=False),
            *reaction_forms(-1/phi, u, r, j, D_reac, D_src),
        ]
        res = sum(terms)
        F_stbl = stabilization_form(
            tau, v, res, h, a, d, r, dt, D_adv, D_diff, D_reac, D_dt, phi, Pv, dx
        )
        forms.append(F_stbl)

    return forms


def steady_advection_diffusion(
    u: Function | FunctionSeries, 
    a: Function | Constant, 
    d: Constant, 
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    tau: str | Callable | None = None,
    Pv: str | Callable | None = None,
    h: GeometricCellQuantity | str = 'hdiam',
) -> list[Form]:
    """
    `𝐚·∇u = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    dx = Measure('dx', u.function_space.mesh)
    forms = [
        *advection_forms(v, u_trial, a, dx=dx, by_parts=False),
        *diffusion_forms(-v, u_trial, d, bcs=bcs, dx=dx, by_parts=True),
        *reaction_forms(-v, u_trial, r, j, dx=dx)
    ]
    if tau is not None:
        terms = [
            *advection_forms(1, u_trial, a, by_parts=False), #TODO bcs here?
            *diffusion_forms(-1, u_trial, d, by_parts=False), #TODO and/or here?
            *reaction_forms(-1, u_trial, r, j),
        ]
        res = sum(terms)
        F_stbl = stabilization_form(tau, v, res, h, a, d, r, Pv=Pv, dx=dx)
        forms.append(F_stbl)

    return forms


def dg_advection_diffusion(
    u: FunctionSeries,
    dt: Constant,
    alpha: Constant | tuple[Constant, Constant],
    a: FunctionSeries,
    d: Function | Expr | Series, 
    D_adv: FiniteDifferenceArgwise,
    D_diff: FiniteDifference,
    D_phi: FiniteDifference = FE,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    *,
    dg_kws: tuple[dict | None, dict | None] = (None, None),
) -> list[Form]:
    return dg_advection_diffusion_reaction(
        u, dt, alpha, a, d,
        D_adv=D_adv, 
        D_diff=D_diff, 
        D_phi=D_phi,
        phi=phi, 
        bcs=bcs,
        dg_kws=dg_kws,
    )


def dg_advection_diffusion_reaction(
    u: FunctionSeries,
    dt: Constant,
    alpha: Constant | tuple[Constant, Constant],
    a,
    d,
    r: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    j: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = FE @ BE,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = FE @ BE,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = FE,
    D_src: FiniteDifference | FiniteDifferenceArgwise = FE,
    D_dt: FiniteDifferenceDerivative = DT,
    D_phi: FiniteDifference = FE,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
    *,
    dg_kws: tuple[dict | None, dict | None] = (None, None),
) -> list[Form]:
    
    phi = D_phi(phi)
    v = TestFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)
    n = FacetNormal(u.function_space.mesh)
    dx = Measure('dx', u.function_space.mesh)
    dS = Measure('dS', u.function_space.mesh)
    if bcs is not None:
        bcs_tuple = bcs.boundary_values(u, 'dirichlet', 'neumann')
    else:
        bcs_tuple = None

    dg_adv_kws, dg_diff_kws = dg_kws
    if dg_adv_kws is None:
        dg_adv_kws = {}
    if dg_diff_kws is None:
        dg_diff_kws = {}

    return [
        derivative_form(v, u, dt, D_dt, dx),
        *dg_advection_forms(v/phi, u, a, n, bcs_tuple, D_adv, dx, dS, **dg_adv_kws),
        *dg_diffusion_forms(-v/phi, u, d, n, h, alpha, bcs_tuple, D_diff, dx, dS, **dg_diff_kws),
        *reaction_forms(-v/phi, u, r, j, D_reac, D_src, dx),
    ]


def dg_steady_advection_diffusion(
    u: Function, 
    alpha: Constant | tuple[Constant, Constant],
    a: Function | Constant, 
    d: Function | Constant,
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    *,
    dg_kws: tuple[dict | None, dict | None] = (None, None),
) -> list[Form]:
    """
    `𝐚·∇u = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)
    n = FacetNormal(u.function_space.mesh)
    dx = Measure('dx', u.function_space.mesh)
    dS = Measure('dS', u.function_space.mesh)
    if bcs is not None:
        bcs_tuple = bcs.boundary_values(u, 'dirichlet', 'neumann')
    else: 
        bcs_tuple = None

    dg_adv_kws, dg_diff_kws = dg_kws
    if dg_adv_kws is None:
        dg_adv_kws = {}
    if dg_diff_kws is None:
        dg_diff_kws = {}

    return [
        *dg_advection_forms(v, u_trial, a, n, bcs_tuple, dx=dx, dS=dS, **dg_adv_kws),
        *dg_diffusion_forms(-v, u_trial, d, n, h, alpha, bcs_tuple, dx=dx, dS=dS, **dg_diff_kws),
        *reaction_forms(-v, u_trial, r, j, dx=dx),
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