from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad,
    inner, grad,
    div, jump, avg, lt, gt, conditional,
)

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, Series, 
    FiniteDifference, FiniteDifferenceArgwise, 
    FE, BE,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions

from .cg_transport import OptionError


def dg_advection_forms(
    v,
    u: FunctionSeries,
    a: Series | Function | Constant | Expr,
    n,
    bcs: BoundaryConditions | tuple | None,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = FE @ BE,
    dx = 1,
    dS = 1,
    dx_opt: int = 0,
    ds_opt: int = 0,
    dS_opt: int = 0,
) -> list[Form]:
    """
    `∫dx v(𝐚·∇u) 
    = - ∫dx (∇v·𝐚)u + ∫dS ... + ∫ds ...`
    """
    if isinstance(D_adv, FiniteDifference):
        D_adv = FE @ D_adv

    solution = u
    D_adv_a, D_adv_u = D_adv
    a = D_adv_a(a, trial=u)
    u = D_adv_u(u, trial=u)

    forms = []

    match dx_opt:
        case 0:
            F_dx = -inner(grad(v), a) * u * dx
        case 1:
            F_dx = -div(v * a) * u * dx
        case _:
            raise OptionError(dx_opt)
    forms.append(F_dx)

    lmbda = conditional(gt(inner(n, a), 0), 1, 0)
    """
    λ = 
    1 if 𝐧·𝐚 > 1 (outflow)
    0 otherwise
    """ 

    na_out = 0.5 * (inner(n, a) + abs(inner(n, a)))
    """
    (𝐧·𝐚)⁺ = 
    if 𝐧·𝐚 > 1 (outflow)
    0 otherwise

    NOTE equivalent to `conditional(gt(inner(n, a), 0), inner(n, a), 0)`
    """

    match dS_opt:
        case 0:
            F_dS = jump(v) * jump(na_out * u) * dS
        case 1:
            F_dS = jump(v) * (na_out('+') * u('+') - na_out('-') * u('-')) * dS
        case 2:
            F_dS = jump(v) * inner(n, a)('+') * conditional(gt(inner(n, a)('+'),  0), u('+'), u('-')) * dS
        case 3:
            F_dS = 2 * jump(v) * avg(na_out * u) * dS
        case 4:
            F_dS = 2 * inner(jump(v, n), avg(lmbda * a * u)) * dS
        case _:
            raise OptionError(dS_opt)
    forms.append(F_dS)

    if bcs is not None:
        ds, u_dirichlet, u_neumann = (
            bcs.boundary_data(solution, 'dirichlet', 'neumann') if isinstance(bcs, BoundaryConditions)
            else bcs
        )
        match ds_opt:
            case 0:
                ds_complement = ds(len(u_dirichlet) + len(u_neumann))
                ds_not_inflow = ds_complement
                if u_neumann:
                    ds_neumann = sum([ds(i) for i, _ in u_neumann[1:]], start=ds(u_neumann[0][0]))
                    ds_not_inflow += ds_neumann
                uI_inflow = lambda uI: conditional(lt(inner(n, a), 0), uI, 0)
                F_inflow = sum([v * inner(n, a) * uI_inflow(uI) * ds(i) for i, uI in u_dirichlet])
                F_outflow = v * inner(n, a) * u * ds_not_inflow
                F_ds = F_inflow + F_outflow
            case 1:
                F_ds = v * lmbda * inner(n, a) * u * ds
                F_ds += sum([v * (1 - lmbda) * inner(n, a) * uD * ds(i) for i, uD in u_dirichlet])
            case _:
                raise OptionError(ds_opt)
        forms.append(F_ds)
    
    return forms
        

def dg_diffusion_forms(
    v,
    u: Function | FunctionSeries,
    d,
    n,
    h,
    alpha: Constant | float | tuple[Constant | float, Constant | float],
    bcs: BoundaryConditions | tuple | None = None,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = FE @ BE,
    dx = 1,
    dS = 1,
) -> list[Form]:
    """
    `∫dx v ∇·(D·∇u)
    = - ∫dx ... + ∫dS ... + ∫ds ...`
    """
    if isinstance(D_diff, FiniteDifference):
        D_diff = FE @ D_diff

    if not isinstance(alpha, tuple):
        alpha = (alpha, alpha)
    alphaI, alphaB = alpha
    if isinstance(alphaI, (float, int)):
        alphaI = Constant(u.function_space.mesh, alphaI)
    if isinstance(alphaB, (float, int)):
        alphaB = Constant(u.function_space.mesh, alphaB)

    D_diff_d, D_diff_u = D_diff
    d = D_diff_d(d, trial=u)
    u = D_diff_u(u, trial=u)

    F_dx = -inner(grad(v), d * grad(u)) * dx

    F_dS = inner(jump(v, n), avg(d * grad(u))) * dS
    F_dS += inner(avg(d * grad(v)), jump(u, n)) * dS
    F_dS += -(alphaI / avg(h)) * inner(jump(v, n), jump(u, n)) * dS

    forms = [F_dx, F_dS]

    if bcs is not None:
        ds, u_dirichlet, u_neumann = (
            bcs.boundary_data(u, 'dirichlet', 'neumann') if isinstance(bcs, BoundaryConditions)
            else bcs
        )
        F_ds = sum([inner(v * n, d * grad(u)) * ds(i) for i, _ in u_dirichlet])
        F_ds += sum([inner(d * grad(v), (u - uD) * n) * ds(i) for i, uD in u_dirichlet])
        F_ds += sum([-(alphaB / h) * v * (u - uD) * ds(i) for i, uD in u_dirichlet])
        F_ds += sum([-v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_ds)

    return forms