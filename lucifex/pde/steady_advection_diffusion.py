from typing import Callable

from ufl import (
    Form, inner, grad, dx, TestFunction, TrialFunction,
    CellDiameter, FacetNormal, dx, dS, 
    inner, grad, div, TestFunction, TrialFunction, 
    avg, jump, conditional, gt,
)

from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions

from .supg import supg_stabilization


def steady_advection_diffusion(
    u: Function, 
    a: Function | Constant, 
    d: Constant, 
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    supg: str | Callable | None = None,
    h: str = 'hdiam',
    gls: bool = False,
) -> list[Form]:
    """
    `𝐚·∇u = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)

    Fa = v * inner(a, grad(u_trial)) * dx
    Fd = inner(grad(v), d * grad(u_trial)) * dx
    forms = [Fa, Fd]

    if r is not None:
        F_reac = -v * r * u_trial * dx
        forms.append(F_reac)
    if j is not None:
        F_src = -v * j * dx 
        forms.append(F_src)

    if bcs is not None:
        ds, u_neumannn = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * uN * ds(i) for i, uN in u_neumannn])
        forms.append(F_neumann)

    if supg is not None:
        res = inner(a, grad(u_trial)) - div(d * grad(u_trial))
        if r is not None:
            res -= r * u_trial
        if j is not None:
            res -= j

        if gls:
            petrov_func = lambda v, a, d, r: (
                inner(grad(v), a) - div(d * grad(v)) - v * (r if r is not None else 0)
            )
        else:
            petrov_func = None
        F_stbl = supg_stabilization(supg, v, res, h, a, d, r, petrov_func=petrov_func)
        forms.append(F_stbl)

    return forms


def steady_advection_diffusion_dg(
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

    outflow = conditional(gt(inner(a, n), 0), 1, 0)
    inflow = 1 - outflow

    match adv_dx:
        case 0:
            F_adv_dx = -inner(grad(v), a) * u_trial * dx
        case 1:
            F_adv_dx = -div(v * a) * u_trial * dx
            
    match adv_dS:
        case 0:
            F_adv_dS = jump(v) * jump(0.5 * (inner(a, n) + abs(inner(a, n))) * u_trial) * dS
        case 1:
            F_adv_dS =  2 * jump(v) * avg (inner(a, n) * u_trial) * dS
        case 2:
            F_adv_dS = jump(v) * inner(a, n)('+') * conditional(inner(a, n)('+') > 0, u_trial('+'), u_trial('-')) * dS

    F_adv_ds = outflow * v * inner(a, n) * u_trial * ds
    F_adv_ds += sum([inflow * v * inner(a, n) * uD * ds(i) for i, uD in u_dirichlet])
    F_adv = F_adv_dx + F_adv_dS + F_adv_ds

    F_diff_dx = inner(grad(v), d * grad(u_trial)) * dx
    F_diff_dS = -inner(jump(v, n), avg(d * grad(u_trial))) * dS
    F_diff_dS -= inner(avg(d * grad(u_trial)), jump(u_trial, n)) * dS
    F_diff_dS += (alpha / avg(h)) * inner(jump(v, n), jump(u_trial, n)) * dS
    F_diff_ds = sum([-inner(d * grad(v), (u_trial - uD) * n) * ds(i) for i, uD in u_dirichlet])
    F_diff_ds += sum([(gamma / h) * v * (u_trial - uD) * ds(i) for i, uD in u_dirichlet])
    F_diff_ds += sum([-v * uN * ds(i) for i, uN in u_neumann])
    F_diff = F_diff_dx + F_diff_dS + F_diff_ds

    forms = [F_adv, F_diff]

    if r is not None:
        F_reac = -v * r * u_trial * dx
        forms.append(F_reac)
    if j is not None:
        F_src = -v * j * dx 
        forms.append(F_src)

    return forms

