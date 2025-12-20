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
    AB1, BE,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions

from .cg import OptionError


def dg_advection_forms(
    v,
    u: FunctionSeries,
    a: Series | Function | Constant | Expr,
    n,
    bcs: BoundaryConditions | None,
    D_adv: FiniteDifferenceArgwise = AB1 @ BE,
    dx = 1,
    dS = 1,
    dx_opt: int = 0,
    dS_opt: int = 0,
) -> list[Form]:
    """
    `∫dx v(𝐚·∇u) 
    = - ∫dx (∇v·𝐚)u + ∫dS ... + ∫ds ...`
    """
    D_adv_a, D_adv_u = D_adv
    a = D_adv_a(a)
    u = D_adv_u(u)

    forms = []

    match dx_opt:
        case 0:
            F_adv_dx = -inner(grad(v), a) * u * dx
        case 1:
            F_adv_dx = -div(v * a) * u * dx
        case _:
            raise OptionError(dx_opt)
        
    forms.append(F_adv_dx)

    an_pos = 0.5 * (inner(n, a) + abs(inner(n, a)))
    # NOTE equivalent to `an_pos = conditional(gt(inner(n, a), 0), inner(n, a), 0)`` 
    match dS_opt:
        case 0:
            F_adv_dS = jump(v) * jump(an_pos * u) * dS
        case 1:
            F_adv_dS = jump(v) * (an_pos('+') * u('+') - an_pos('-') * u('-')) * dS
        case 2:
            F_adv_dS = jump(v) * inner(n, a)('+') * conditional(gt(inner(n, a)('+'),  0), u('+'), u('-')) * dS
        case 3:
            F_adv_dS = 2 * jump(v) * avg(an_pos * u) * dS
        case _:
            raise OptionError(dS_opt)

    forms.append(F_adv_dS)

    if bcs is not None:
        ds, u_dirichlet = bcs.boundary_data(u.function_space, 'dirichlet')
        ds_complement = ds(len(u_dirichlet))
        uI_or_trial = lambda uI: conditional(lt(inner(n, a), 0), uI, u)
        F_inflow = sum([v * inner(n, a) * uI_or_trial(uI) * ds(i) for i, uI in u_dirichlet])
        F_outflow = v * inner(n, a) * u * ds_complement
        forms.extend([F_inflow, F_outflow])

    return forms
        

def dg_diffusion_forms(
    v,
    n,
    a,
    d,
    u,
    opt: int = 0,
) -> list[Form]:
    ...
    # F_diff_dx = inner(grad(v / phi), d * grad(D_diff(u))) * dx
    # F_diff_dS = -inner(jump(v / phi, n), avg(d * grad(D_diff(u)))) * dS
    # F_diff_dS += -inner(avg(d * grad(v / phi)), jump(D_diff(u), n)) * dS
    # F_diff_dS += (alpha / avg(h)) * inner(jump(v / phi, n), jump(D_diff(u), n)) * dS
    # F_diff_ds = sum([-inner(d * grad(v / phi), (D_diff(u) - uD) * n) * ds(i) for i, uD in u_dirichlet])
    # F_diff_ds += sum([-inner(v * n / phi, d * grad(D_diff(u))) * ds(i) for i, uD in u_dirichlet])
    # F_diff_ds += sum([(gamma / h) * v * (D_diff(u) - uD) * ds(i) for i, uD in u_dirichlet])
    # F_diff_ds += sum([-(v / phi) * uN * ds(i) for i, uN in u_neumann])
    # F_diff = F_diff_dx + F_diff_dS + F_diff_ds
            






# from steady

# outflow = conditional(gt(inner(n, a), 0), 1, 0)
# inflow = 1 - outflow

# match adv_dx:
#     case 0:
#         F_adv_dx = -inner(grad(v), a) * u_trial * dx
#     case 1:
#         F_adv_dx = -div(v * a) * u_trial * dx
        
# match adv_dS:
#     case 0:
#         F_adv_dS = jump(v) * jump(0.5 * (inner(n, a) + abs(inner(n, a))) * u_trial) * dS
#     case 1:
#         F_adv_dS =  2 * jump(v) * avg (inner(n, a) * u_trial) * dS
#     case 2:
#         F_adv_dS = jump(v) * inner(n, a)('+') * conditional(inner(n, a)('+') > 0, u_trial('+'), u_trial('-')) * dS

# F_adv_ds = outflow * v * inner(n, a) * u_trial * ds
# F_adv_ds += sum([inflow * v * inner(n, a) * uD * ds(i) for i, uD in u_dirichlet])
# F_adv = F_adv_dx + F_adv_dS + F_adv_ds

# F_diff_dx = inner(grad(v), d * grad(u_trial)) * dx
# F_diff_dS = -inner(jump(v, n), avg(d * grad(u_trial))) * dS
# F_diff_dS -= inner(avg(d * grad(u_trial)), jump(u_trial, n)) * dS
# F_diff_dS += (alpha / avg(h)) * inner(jump(v, n), jump(u_trial, n)) * dS
# F_diff_ds = sum([-inner(d * grad(v), (u_trial - uD) * n) * ds(i) for i, uD in u_dirichlet])
# F_diff_ds += sum([(gamma / h) * v * (u_trial - uD) * ds(i) for i, uD in u_dirichlet])
# F_diff_ds += sum([-v * uN * ds(i) for i, uN in u_neumann])
# F_diff = F_diff_dx + F_diff_dS + F_diff_ds

# forms = [F_adv, F_diff]

# if r is not None:
#     F_reac = -v * r * u_trial * dx
#     forms.append(F_reac)
# if j is not None:
#     F_src = -v * j * dx 
#     forms.append(F_src)

# return forms