from ufl.core.expr import Expr
from ufl import (Measure, Form, FacetNormal, inner, grad, inv, div,
                 Dx, TrialFunction, TestFunction, as_vector,
                 det, transpose, TrialFunctions, TestFunctions)
from dolfinx.fem import FunctionSpace

from lucifex.fem import Function, Constant
from lucifex.fdm import FunctionSeries

from lucifex.solver import BoundaryConditions
from lucifex.utils import is_tensor


def darcy_streamfunction(
    psi: Function | FunctionSeries,
    k: Expr | Function | Constant | float,
    mu: Expr | Function | Constant | float,
    fx: Expr | Function | Constant | float | None = None,
    fy: Expr | Function | Constant | float | None = None,
) -> list[Form]:
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
    bcs: BoundaryConditions | None = None,
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

    if bcs is not None:
        ds, p_natural = bcs.boundary_data(up, 'natural')
        F_bcs = sum([inner(v, n) * pN * ds(i) for i, pN in p_natural])
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
    if isinstance(p, FunctionSpace):
        fs = p
    else:
        fs = p.function_space
    dx = Measure('dx', fs.mesh)
    v = TestFunction(fs)
    p_trial = TrialFunction(fs)
    
    F_lhs = -inner(grad(v), (k / mu) * grad(p_trial)) * dx
    forms = [F_lhs]

    if f is not None:
        if isinstance(f, tuple):
            f = as_vector(f)
        F_rhs = -v * div((k / mu) * f) * dx
        forms.append(F_rhs)

    if bcs is not None:
        ds, p_neumann, p_robin = bcs.boundary_data(p, 'neumann', 'robin')
        if p_neumann:
            F_neumann = sum([v * pN * ds(i) for i, pN in p_neumann])
            forms.append(F_neumann)
        if p_robin:
            F_robin = sum([v * pR * ds(i) for i, pR in p_robin])
            forms.append(F_robin)

    if f is None and bcs is None:
        F_zero = v * Constant(fs.mesh, 0.0) * dx
        forms.append(F_zero)

    return forms


def darcy_velocity_from_pressure(
    p: Function | Expr,
    k: Function | Expr | Constant,
    mu: Function | Expr | Constant,
    f: Expr | tuple,
) -> Expr:
    """
    `𝐮 = -(K/μ)⋅(∇p - 𝐟)`
    """
    if isinstance(f, tuple):
        f = as_vector(f)
    return -(k / mu) * (grad(p) - f)