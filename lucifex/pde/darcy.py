from ufl.core.expr import Expr
from ufl import (
    Measure, Form, FacetNormal, inner, grad, inv, div,
    Dx, as_vector, det, transpose, TrialFunctions, TestFunctions,
)

from lucifex.fem import Function, Constant
from lucifex.fdm import FunctionSeries
from lucifex.solver import BoundaryConditions, BlockedForm
from lucifex.utils.fenicsx_utils import is_tensor, is_zero

from .poisson import poisson


def darcy_streamfunction(
    psi: Function | FunctionSeries,
    k: Expr | Function | Constant | float,
    mu: Expr | Function | Constant | float,
    fx: Expr | Function | Constant | float | None = None,
    fy: Expr | Function | Constant | float | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇·(μKᵀ·∇ψ / det(K)) = - ∂fʸ/∂x + ∂fˣ/∂y`

    for tensor-valued `K` or

    `∇·(μ·∇ψ / K) = - ∂fʸ/∂x + ∂fˣ/∂y`

    for scalar-valued `K`.
    """

    if is_tensor(k):
        weight = (mu / det(k)) * transpose(k)
    else:
        weight = (mu / k)

    rhs = 0
    if not is_zero(fx):
        rhs += Dx(fx, 1)
    if not is_zero(fy):
        rhs += -Dx(fy, 0)

    return poisson(psi, rhs, weight, bcs)


def darcy_incompressible(
    up: FunctionSeries,
    k: Expr | Function | Constant,
    mu: Expr | Function | Constant,
    f: Expr | None = None,
    s: Expr | None = None,
    bcs: BoundaryConditions | None = None,
    blocked: bool = False,
) -> list[Form] | BlockedForm:
    """
    `∇⋅𝐮 = s` \\
    `𝐮 = -(K/μ)⋅(∇p - 𝐟)`
    
    `F(𝐮,p;𝐯,q) = ∫ q(∇·𝐮) dx ` \\
    `+ ∫ 𝐯·(μ K⁻¹⋅𝐮) dx - ∫ p(∇·𝐯) dx - ∫ 𝐯·𝐟 dx + ∫ p(𝐯·𝐧) ds`
    """
    dx = Measure('dx', up.function_space.mesh)
    v, q = TestFunctions(up.function_space)
    u, p = TrialFunctions(up.function_space)
    n = FacetNormal(up.function_space.mesh)

    F_div = q * div(u) * dx

    F_src = 0
    if s is not None:
        F_src = -q * s * dx

    if is_tensor(k):
        F_velocity = inner(v, mu * inv(k) * u) * dx
    else:
        F_velocity = inner(v, mu * u / k) * dx
    F_pressure = -p * div(v) * dx

    F_force = 0
    if f is not None:
        F_force = -inner(v, f) * dx

    F_bcs = 0
    if bcs is not None:
        ds, p_natural = bcs.boundary_data(up, 'natural')
        F_bcs = sum([inner(v, n) * pN * ds(i) for i, pN in p_natural])

    if blocked:
        return BlockedForm(
            [F_velocity + F_force + F_bcs, F_pressure], 
            [F_div + F_src, None],
        )
    else:
        forms = [F_div, F_velocity, F_pressure]
        if F_src:
            forms.append(F_src)
        if F_force:
            forms.append(F_force)
        if F_bcs:
            forms.append(F_bcs)
        return forms


def darcy_pressure(
    p: Function | FunctionSeries,
    k: Expr | Function | Constant | float,
    mu: Expr | Function | Constant | float,
    f: Expr | Function | Constant | float | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇⋅((K/μ)⋅∇p) = ∇⋅((K/μ)⋅𝐟)`
    """
    if f is not None:
        if isinstance(f, tuple):
            f = as_vector(f)
        rhs = div((k / mu) * f)
    else:
        rhs = None

    weight = (k / mu)
    return poisson(p, rhs, weight, bcs)


def darcy_velocity_from_pressure(
    p: Function | Expr,
    k: Function | Expr | Constant,
    mu: Function | Expr | Constant,
    f: Expr | tuple | None = None,
) -> Expr:
    """
    `𝐮 = -(K/μ)⋅(∇p - 𝐟)`
    """
    u = -(k / mu) * grad(p)

    if f is not None:
        if isinstance(f, tuple):
            f = as_vector(f)
        u += (k / mu) * f

    return u