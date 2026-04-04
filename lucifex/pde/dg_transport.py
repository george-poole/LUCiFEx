from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, Argument,
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
from lucifex.utils.npy_utils import AnyFloat
from lucifex.utils.fenicsx_utils import extract_function_space

from .cg_transport import OptionError


def dg_advection_forms(
    v: Argument,
    u: FunctionSeries,
    a: Series | Function | Constant | Expr,
    n,
    bcs: BoundaryConditions | tuple | None,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = FE @ BE,
    dx = 1,
    dS = 1,
    *,
    dx_opt: int = 0,
    ds_opt: int = 0,
    dS_opt: int = 0,
    D_ds: FiniteDifference | None = None,
    D_dS: FiniteDifference | None = None,
) -> list[Form]:
    """
    `∫dx v(𝐚·∇u) 
    = - ∫dx (∇v·𝐚)u + ∫dS ... + ∫ds ...`
    """
    if isinstance(D_adv, FiniteDifference):
        D_adv = FE @ D_adv

    D_adv_a, D_adv_u = D_adv
    Da = D_adv_a(a, trial=u)
    Du = D_adv_u(u, trial=u)

    forms = []

    match dx_opt:
        case 0:
            F_dx = -inner(grad(v), Da) * Du * dx
        case 1:
            F_dx = -div(v * Da) * Du * dx
        case _:
            raise OptionError(dx_opt)
    forms.append(F_dx)

    lmbda = conditional(gt(inner(n, Da), 0), 1, 0)
    """
    λ = 
    1 if 𝐧·𝐚 > 1 (outflow)
    0 otherwise
    """ 

    na_out = 0.5 * (inner(n, Da) + abs(inner(n, Da)))
    """
    (𝐧·𝐚)⁺ = 
    if 𝐧·𝐚 > 1 (outflow)
    0 otherwise

    Equivalent to `conditional(gt(inner(n, a), 0), inner(n, a), 0)`
    """

    _Du = D_dS(u) if D_dS is not None else Du
    match dS_opt:
        case 0:
            F_dS = jump(v) * jump(na_out * _Du) * dS
        case 1:
            F_dS = jump(v) * (na_out('+') * _Du('+') - na_out('-') * _Du('-')) * dS
        case 2:
            F_dS = jump(v) * inner(n, Da)('+') * conditional(gt(inner(n, Da)('+'),  0), _Du('+'), _Du('-')) * dS
        case 3:
            F_dS = 2 * jump(v) * avg(na_out * _Du) * dS
        case 4:
            F_dS = 2 * inner(jump(v, n), avg(lmbda * Da * _Du)) * dS
        case _:
            raise OptionError(dS_opt)
    forms.append(F_dS)

    if bcs is not None:
        ds, u_dirichlet, u_neumann = (
            bcs.boundary_data(u, 'dirichlet', 'neumann') if isinstance(bcs, BoundaryConditions)
            else bcs
        )
        _Du = D_ds(u) if D_ds is not None else Du
        match ds_opt:
            case 0:
                ds_complement = ds(len(u_dirichlet) + len(u_neumann))
                ds_not_inflow = ds_complement
                if u_neumann:
                    ds_not_inflow = sum([ds(i) for i, _ in u_neumann], start=ds_not_inflow)
                uI_inflow = lambda uI: conditional(lt(inner(n, Da), 0), uI, 0)
                F_inflow = sum([v * inner(n, Da) * uI_inflow(uI) * ds(i) for i, uI in u_dirichlet])
                F_outflow = v * inner(n, Da) * _Du * ds_not_inflow
                F_ds = F_inflow + F_outflow
            case 1:
                F_ds = v * lmbda * inner(n, Da) * _Du * ds
                F_ds += sum([v * (1 - lmbda) * inner(n, Da) * uD * ds(i) for i, uD in u_dirichlet])
            case _:
                raise OptionError(ds_opt)
        forms.append(F_ds)
    
    return forms
        

def dg_diffusion_forms(
    v,
    u: Function | FunctionSeries | Argument,
    d,
    n,
    h,
    alpha: Constant | float | tuple[Constant | float, Constant | float],
    bcs: BoundaryConditions | tuple | None = None,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = FE @ BE,
    dx = 1,
    dS = 1,
    *,
    D_ds: FiniteDifference | None = None,
    D_dS: FiniteDifference | None = None,
) -> list[Form]:
    """
    `∫dx v ∇·(D·∇u)
    = - ∫dx ... + ∫dS ... + ∫ds ...`
    """
    if isinstance(D_diff, FiniteDifference):
        D_diff = FE @ D_diff

    fs = extract_function_space(u)
    if not isinstance(alpha, tuple):
        alpha = (alpha, alpha)
    alphaI, alphaB = alpha
    if isinstance(alphaI, AnyFloat):
        alphaI = Constant(fs.mesh, alphaI)
    if isinstance(alphaB, AnyFloat):
        alphaB = Constant(fs.mesh, alphaB)

    D_diff_d, D_diff_u = D_diff
    Dd = D_diff_d(d, trial=u)
    Du = D_diff_u(u, trial=u)

    F_dx = -inner(grad(v), Dd * grad(Du)) * dx

    _Du = D_dS(u) if D_dS is not None else Du
    F_dS = inner(jump(v, n), avg(Dd * grad(_Du))) * dS
    F_dS += inner(avg(Dd * grad(v)), jump(_Du, n)) * dS
    F_dS += -(alphaI / avg(h)) * inner(jump(v, n), jump(_Du, n)) * dS

    forms = [F_dx, F_dS]

    if bcs is not None:
        ds, u_dirichlet, u_neumann = (
            bcs.boundary_data(u, 'dirichlet', 'neumann') if isinstance(bcs, BoundaryConditions)
            else bcs
        )
        _Du = D_ds(u) if D_ds is not None else Du
        F_ds = sum([inner(v * n, Dd * grad(_Du)) * ds(i) for i, _ in u_dirichlet])
        F_ds += sum([inner(Dd * grad(v), (_Du - uD) * n) * ds(i) for i, uD in u_dirichlet])
        F_ds += sum([-(alphaB / h) * v * (_Du - uD) * ds(i) for i, uD in u_dirichlet])
        F_ds += sum([-v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_ds)

    return forms