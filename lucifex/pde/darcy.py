from ufl.core.expr import Expr
from ufl import (Measure, Form, FacetNormal, inner, inv, div,
                 Dx, TrialFunction, TestFunction,
                 det, transpose, TrialFunctions, TestFunctions)

from lucifex.fem import Function, Constant
from lucifex.fdm import FunctionSeries

from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import BoundaryConditions
from lucifex.utils import is_tensor


def darcy_streamfunction(
    psi: FunctionSeries,
    k: Expr | Function | Constant | float,
    mu: Expr | Function | Constant | float,
    fx: Expr | Function | Constant | float | None = None,
    fy: Expr | Function | Constant | float | None = None,
) -> tuple[Form, Form]:
    """
    `∇·(μKᵀ·∇ψ / det(K)) = ∂fʸ/∂x - ∂fˣ/∂y`

    for tensor-valued `K` or

    `∇·(μ·∇ψ / K) = ∂fʸ/∂x - ∂fˣ/∂y`

    for scalar-valued `K`.
    """
    _none = (None, 0) 
    dx = Measure('dx', psi.function_space.mesh)
    v = TestFunction(psi.function_space)
    psi_trial = TrialFunction(psi.function_space)
    if is_tensor(k):
        F_lhs = -(mu / det(k)) * inner(grad(v), transpose(k) * grad(psi_trial)) * dx 
    else:
        F_lhs = -(mu / k) * inner(grad(v), grad(psi_trial)) * dx
    forms = [F_lhs]
    if not fx in _none:
        F_egx = -v * Dx(fx, 1) * dx
        forms.append(F_egx)
    if not fy in _none:
        F_egy = v * Dx(fy, 0) * dx
        forms.append(F_egy)
    if fx in _none and fy in _none:
        F_zero = v * Constant(psi.function_space.mesh, 0.0) * dx
        forms.append(F_zero)
    return forms


def darcy_incompressible(
    up: FunctionSeries,
    k: Expr | Function | Constant,
    mu: Expr | Function | Constant,
    f: Expr | None = None,
    p_bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇⋅𝐮 = 0` \\
    `𝐮 = -(K/μ)⋅(∇p - 𝐟)`
    
    `F(𝐮,p;𝐯,q) = ∫ q(∇·𝐮) dx ` \\
    `+ ∫ 𝐯·(μ K⁻¹⋅𝐮) dx - ∫ p(∇·𝐯) dx - ∫ 𝐯·𝐟 dx + ∫ p(𝐯·𝐧) ds`
    """
    dx = Measure('dx', up.function_space.mesh)
    v, q = TestFunctions(up.function_space)
    u, p = TrialFunctions(up.function_space)
    n = FacetNormal(up.function_space.mesh)

    F_div = q * div(u) * dx
    if is_tensor(k):
        F_velocity = inner(v, mu * inv(k) * u) * dx
    else:
        F_velocity = inner(v, mu * u / k) * dx
    F_pressure = -p * div(v) * dx

    forms = [F_div, F_velocity, F_pressure]

    if f is not None:
        F_buoyancy = inner(v, f) * dx
        forms.append(F_buoyancy)

    if p_bcs is not None:
        ds, p_natural = p_bcs.boundary_data(up.function_space, 'natural')
        F_bcs = sum([inner(v, n) * pN * ds(i) for i, pN in p_natural])
        forms.append(F_bcs)

    return forms