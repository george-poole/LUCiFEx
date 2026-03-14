from ufl.core.expr import Expr
from ufl import (
    Measure, Form, FacetNormal, inner, grad, inv, div,
    Dx, as_vector, det, transpose, TrialFunctions, TestFunctions,
)

from lucifex.fem import Function, Constant
from lucifex.fdm import FunctionSeries
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import is_tensor, is_none, BlockedForm, create_zero_form

from .poisson import poisson


def darcy(
    up: FunctionSeries,
    k: Expr | Function | Constant,
    mu: Expr | Function | Constant,
    f: Expr | None = None,
    s: Expr | None = None,
    bcs: BoundaryConditions | None = None,
    *,
    add_zero: tuple[bool | None, bool | None] = (False, False),
    blocked: bool = False,
) -> list[Form] | BlockedForm:
    """
    `∇⋅𝐮 = s` \\
    `𝐮 = -(K/μ)⋅(∇p - 𝐟)`
    
    `F(𝐮,p;𝐯,q) = ∫ q(∇·𝐮)dx - ∫ qs dx ` \\
    `+ ∫ 𝐯·(μ K⁻¹⋅𝐮) dx - ∫ p(∇·𝐯) dx - ∫ 𝐯·𝐟 dx + ∫ p(𝐯·𝐧) ds`
    """
    add_v_zero, add_q_zero = add_zero
    if add_v_zero is None:
        add_v_zero = f is None
    if add_q_zero is None:
        add_q_zero = s is None

    dx = Measure('dx', up.function_space.mesh)
    n = FacetNormal(up.function_space.mesh)
    v, q = TestFunctions(up.function_space)
    u, p = TrialFunctions(up.function_space)
    
    F_div = q * div(u) * dx

    F_src = 0
    if s is not None:
        F_src = -q * s * dx
    if add_q_zero:
        F_src = create_zero_form(q, up.function_space.mesh, dx)

    if is_tensor(k):
        F_velocity = inner(v, mu * inv(k) * u) * dx
    else:
        F_velocity = inner(v, mu * u / k) * dx
    F_pressure = -p * div(v) * dx

    F_force = 0
    if f is not None:
        F_force = -inner(v, f) * dx
    if add_v_zero:
        F_force = create_zero_form(v, up.function_space.mesh, dx, v.ufl_shape)

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

    with tensor-valued `K` or

    `∇·(μ·∇ψ / K) = - ∂fʸ/∂x + ∂fˣ/∂y`

    with scalar-valued `K` for an incompressible flow  `∇⋅𝐮 = 0`.
    """

    if is_tensor(k):
        weight = (mu / det(k)) * transpose(k)
    else:
        weight = (mu / k)

    rhs = 0
    if not is_none(fx):
        rhs += Dx(fx, 1)
    if not is_none(fy):
        rhs += -Dx(fy, 0)

    return poisson(psi, rhs, weight, bcs)


def darcy_pressure(
    p: Function | FunctionSeries,
    k: Expr | Function | Constant | float,
    mu: Expr | Function | Constant | float,
    f: Expr | Function | Constant | float | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇⋅((K/μ)⋅∇p) = ∇⋅((K/μ)⋅𝐟)` for an incompressible flow  `∇⋅𝐮 = 0`.
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