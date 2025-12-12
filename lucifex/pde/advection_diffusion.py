from typing import Callable

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
    a: FunctionSeries,
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
    if isinstance(d, Series):
        d = D_disp(d)
    if isinstance(phi, Series):
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
        ds, c_neumann = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * (1/phi) * cN * ds(i) for i, cN in c_neumann])
        forms.append(F_neumann)

    if supg is not None:
        res = dudt + adv + diff
        F_supg = supg_stabilization(supg, v, res, h, a, d, dt=dt, D_adv=D_adv, D_diff=D_diff, phi=phi)
        forms.append(F_supg)

    return forms


def advection_diffusion_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: FunctionSeries,
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
    if isinstance(d, Series):
        d = D_disp(d)
    if isinstance(phi, Series):
        phi = D_phi(phi)
        
    forms = advection_diffusion(
        u, dt, a, d, D_adv, D_diff, D_disp, D_phi, phi, bcs, supg=None,
    )
    v = TestFunction(u.function_space)

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
) -> list[Form]:
    if bcs is None:
        bcs = BoundaryConditions()
    ds, c_dirichlet, c_neumann = bcs.boundary_data(u.function_space, 'dirichlet', 'neumann')

    if isinstance(phi, Series):
        phi = D_phi(phi)
    if isinstance(d, Series):
        d = D_phi(d)

    v = TestFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)
    n = FacetNormal(u.function_space.mesh)
    if isinstance(alpha, float):
        alpha = Constant(u.function_space.mesh, alpha)
    if isinstance(beta, float):
        beta = Constant(u.function_space.mesh, beta)

    F_dcdt = v * DT(u, dt) * dx

    Pe = ... # FIXME

    D_adv_u, D_adv_c = D_adv
    uEff = (1/phi) * (D_adv_u(a) - (1/Pe) * grad(phi))
    cAdv = D_adv_c(u)
    outflow = conditional(gt(dot(uEff, n), 0), 1, 0)

    F_adv_dx = -inner(grad(v), uEff * cAdv) * dx
    F_adv_dS = 2 * inner(jump(v, n), avg(outflow * uEff * cAdv)) * dS
    F_adv_ds = inner(v, outflow * inner(uEff, n) * cAdv) * ds 
    F_adv_ds += sum([inner(v, (1 - outflow) * inner(uEff, n) * cD) * ds(i) for cD, i in c_dirichlet])
    #### alternatice c.f. wells, burkardt
    # un = 0.5 * (inner(u, n) + abs(inner(u, n)))
    # un = outflow * inner(u, n)
    # F_adv_dS = inner(jump(v), un('+') * c('+') - un('-') * c('-')) * dS
    # F_adv_dS = inner(jump(v), jump(un * c)) * dS
    # F_adv_ds = v * (un * c + ud * cD) * ds(i)   # NOTE this applies DiricletBC on inflow only?
    ####
    F_adv = F_adv_dx + F_adv_dS + F_adv_ds

    cDiff = D_diff(u)
    # + ∫ ∇v⋅∇c dx
    F_diff_dx = inner(grad(v), grad(cDiff)) * dx
    # - ∫ [vn]⋅{∇c} dS
    F_diff_dS = -inner(jump(v, n), avg(grad(cDiff))) * dS
    # - ∫ [vn]⋅{∇c} dS
    F_diff_dS += -inner(avg(grad(v)), jump(cDiff, n)) * dS
    # + ∫ (α / h)[vn]⋅[cn] dS
    F_diff_dS += (alpha / avg(h)) * inner(jump(v, n), jump(cDiff, n)) * dS # TODO h('+') or avg(h) ?
    # ...
    F_diff_ds = sum([-(inner(grad(v), (cDiff - cD) * n) + inner(v * n, grad(cDiff))) * ds(i) for cD, i in c_dirichlet])
    F_diff_ds += sum([(gamma / h) * v * (cDiff - cD) * ds(i) for cD, i in c_dirichlet])
    F_diff_ds += sum([-v * cN * ds(i) for cN, i in c_neumann])
    F_diff = (1/Pe) * (F_diff_dx + F_diff_dS + F_diff_ds)

    return [F_dcdt, F_adv, F_diff]

# TODO debug, debug, debug
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
    
    forms = advection_diffusion_dg(u, dt, phi, a, d, D_adv, D_diff, D_phi, alpha, gamma, bcs)

    # if np.isclose(float(Da), 0):
    #     return forms
    
    if isinstance(phi, Series):
        phi = D_phi(phi)

    r = D_reac(r, trial=u)
    v = TestFunction(u.function_space)
    F_reac = -v  * (r / phi) * dx
    forms.append(F_reac)

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
