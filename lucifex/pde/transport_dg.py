from ufl import (dx, dS, Form, CellDiameter, FacetNormal,
                 TestFunction,
                 jump, avg, dot, conditional, gt)
from ufl.geometry import CellDiameter
from ufl.core.expr import Expr

from lucifex.solver import BoundaryConditions
from lucifex.fem import Function, Constant
from lucifex.fdm import (DT, AB1, FiniteDifference, FunctionSeries, Series, 
                        apply_finite_difference)
from lucifex.fdm.ufl_operators import inner, grad


# TODO debug and test
def advection_diffusion_dg(
    u: FunctionSeries,
    dt: Constant,
    a: FunctionSeries,
    d: Function | Expr | Series, 
    alpha: Constant | float,
    gamma: Constant | float,
    D_adv: tuple[FiniteDifference, FiniteDifference],
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


def advection_diffusion_reaction_dg(
    u: FunctionSeries,
    dt: Constant,
    a,
    d,
    r,
    alpha: float,
    gamma: float,
    D_adv: tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    D_reac: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    
    forms = advection_diffusion_dg(u, dt, phi, a, d, D_adv, D_diff, D_phi, alpha, gamma, bcs)

    # if np.isclose(float(Da), 0):
    #     return forms
    
    if isinstance(phi, Series):
        phi = D_phi(phi)

    r = apply_finite_difference(D_reac, r, u)
    v = TestFunction(u.function_space)
    F_reac = -v  * (r / phi) * dx
    forms.append(F_reac)

    return forms



