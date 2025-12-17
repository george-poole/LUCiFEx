from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, dx, TestFunction, TrialFunction,
    inner, grad, TestFunction, TrialFunction,
)


from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, FiniteDifference, FiniteDifferenceArgwise, 
    DT, AB1,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions


def diffusion(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_diff: FiniteDifference, # TODO tuple
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t = ∇·(D·∇u)`

    `(uⁿ⁺¹ - uⁿ) / Δtⁿ = ∇·(𝒟(D·∇u))`
    """
    v = TestFunction(u.function_space)
    Ft = v * DT(u, dt) * dx
    Fd = inner(grad(v), d * grad(D_diff(u))) * dx
    forms = [Ft, Fd]
    if bcs is not None:
        ds, u_neumann = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([-v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)
    return forms


def diffusion_reaction(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_diff: FiniteDifference,
    r: Function | Constant | None = None,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t = ∇·(D·∇u) + R`

    `(uⁿ⁺¹ - uⁿ) / Δtⁿ = ∇·(𝒟(D·∇u)) + 𝒟(R)`
    """
    forms = diffusion(u, dt, d, D_diff, bcs)
    if r is not None:
        v = TestFunction(u.function_space)
        F_reac = -v * D_reac(r) * dx
        forms.append(F_reac)
    return forms


def steady_diffusion_reaction(
    u: Function,
    d: Constant,
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    zero: bool = True,
) -> tuple[Form, Form]:
    """"
    `∇·(D·∇u) = Ru + J`
    """
    u_trial = TrialFunction(u.function_space)
    v = TestFunction(u.function_space)
    Fd =  -inner(grad(v), d * grad(u_trial)) * dx
    forms = [Fd]
    if r is not None:
        Fr = -v * r * u_trial * dx
        forms.append(Fr)
    if j is not None:
        Fs = -v * j * dx
        forms.append(Fs)
    if zero:
        Fzero = v * Constant(u.function_space.mesh, 0.0) * dx
        forms.append(Fzero)
    return forms

