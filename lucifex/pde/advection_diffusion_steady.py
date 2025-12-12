from typing import Callable

from ufl import (
    Form, inner, grad, dx, TestFunction, TrialFunction,
    CellDiameter, FacetNormal, dx, dS, 
    inner, grad, div, TestFunction, TrialFunction, 
    avg, jump,
)

from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions

from .supg import supg_stabilization


def advection_diffusion_steady(
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
        Fr = -v * r * u_trial * dx
        forms.append(Fr)
    if j is not None:
        Fs = -v * j * dx 
        forms.append(Fs)

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


def advection_diffusion_steady_dg(
    u: Function, 
    a: Function | Constant, 
    d: Constant, 
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
):
    """
    `𝐚·∇u = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)
    n = FacetNormal(u.function_space.mesh)

    aOut = 0.5 * (inner(a, n) + abs(inner(a, n)))
    aIn = 0.5 * (inner(a, n) - abs(inner(a, n)))

    Fa = inner(jump(v), aOut('+') * u('+') - aIn('-') * u('-')) * dS
    Fd = d * inner(jump(v), jump(u_trial)) / avg(h)
    forms = [Fa, Fd]
    if r is not None:
        Fr = v * r * dx
        forms.append(Fr)
    if j is not None:
        Fr = -v * j * dx 
        forms.append(Fr)
    if bcs is not None:
        ds, u_dirichlet = bcs.boundary_data(u.function_space, 'dirichlet')
        raise NotImplementedError

    return forms

