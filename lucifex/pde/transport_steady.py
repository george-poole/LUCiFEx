from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, dx, TestFunction, TrialFunction,
    CellDiameter, FacetNormal, dx, dS, 
    inner, grad, div, TestFunction, TrialFunction, Dx, 
    avg, sqrt, tanh, jump,
)

from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions

from .supg import peclet


def reaction_diffusion(
    u: Function,
    d: Constant,
    r: Function | Constant | None = None,
    s: Function | Constant | None = None,
    zero: bool = True,
) -> tuple[Form, Form]:
    """"
    ∇·(D·∇u) = ru + s
    """
    u_trial = TrialFunction(u.function_space)
    v = TestFunction(u.function_space)
    Fd =  -inner(grad(v), d * grad(u_trial)) * dx
    forms = [Fd]
    if r is not None:
        Fr = -v * r * u_trial * dx
        forms.append(Fr)
    if s is not None:
        Fs = -v * s * dx
        forms.append(Fs)
    if zero:
        Fzero = v * Constant(u.function_space.mesh, 0.0) * dx
        forms.append(Fzero)
    return forms


def advection_diffusion_steady(
    u: Function, 
    a: Function | Constant, 
    d: Constant, 
    r: Function | Constant | None = None,
    s: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    stabilization: str | None = None,
) -> list[Form]:
    """
    `𝐚·∇u = ∇·(D·∇u) + ru + s`
    """
    stabilization = str(stabilization).lower()
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    h = CellDiameter(u.function_space.mesh)

    Fa = v * inner(a, grad(u_trial)) * dx
    Fd = inner(grad(v), d * grad(u_trial)) * dx
    forms = [Fa, Fd]

    if r is not None:
        Fr = -v * r * u_trial * dx
        forms.append(Fr)
    if s is not None:
        Fs = -v * s * dx 
        forms.append(Fs)

    if bcs is not None:
        ds, u_neumannn = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * uN * ds(i) for i, uN in u_neumannn])
        forms.append(F_neumann)

    if stabilization in ('supg', 'gls'):
        res = inner(a, grad(u_trial)) - div(d * grad(u_trial))
        if r is not None:
            res -= r * u_trial
        if s is not None:
            res -= s
        Pe = peclet(h, a, d)
        beta = (1 / tanh(Pe)) -  (1 / Pe)
        tau = 0.5 * beta * h / sqrt(inner(a, a))
        if stabilization == 'supg':
            Pv = inner(a, grad(v))
        if stabilization == 'gls':
            Pv = inner(a, grad(v)) - div(d * grad(v))
            if r is not None:
                Pv -= r * v
        Fsupg = tau * Pv * res * dx
        forms.append(Fsupg)
        return forms

    if stabilization.startswith('su'):
        assert u.function_space.mesh.geometry.dim == 1
        if stabilization.endswith('upwind'):
            beta = Constant(u.function_space.mesh, 1.0)
        elif stabilization.endswith('optimal'):
            Pe = peclet(h, a, d)
            beta = (1 / tanh(Pe)) -  (1 / Pe)
        else:
            raise ValueError(f"'{stabilization}' not recognised.")
        Fsu = 0.5 * beta * h * Dx(v, 0) * inner(a, grad(u_trial)) * dx
        forms.append(Fsu)
        return forms

    return forms


def advection_diffusion_steady_dg(
    u: Function, 
    a: Function | Constant, 
    d: Constant, 
    r: Function | Constant | None = None,
    s: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
):
    """
    `𝐚·∇u = ∇·(D·∇u) + ru + s`
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
    if s is not None:
        Fr = -v * s * dx 
        forms.append(Fr)
    if bcs is not None:
        ds, u_dirichlet = bcs.boundary_data(u.function_space, 'dirichlet')
        raise NotImplementedError

    return forms

