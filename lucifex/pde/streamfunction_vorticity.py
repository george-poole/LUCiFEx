from ufl.core.expr import Expr
from ufl import Form, as_matrix, as_vector, grad, curl, Dx, split

from lucifex.fem import Function
from lucifex.fdm.ufl_overloads import unary_overload
from lucifex.solver import BoundaryConditions
from .poisson import poisson


@unary_overload
def velocity_from_streamfunction(
    psi: Function,
    neg: bool = False,
) -> Expr:
    """
    `𝐮 = ((0, ±1), (∓1, 0))·∇ψ` depending on sign convention 
    for an incompressible flow `∇·𝐮 = 0`.
    """
    if neg:
        mat = ([0, -1], [1, 0])
    else:
        mat = ([0, 1], [-1, 0])
    return as_matrix(mat) * grad(psi)


@unary_overload
def vorticity_from_velocity(
    u: Function | Expr | tuple[Function | Expr, ...],
    d2: bool = False,
) -> Expr:
    """
    `𝝎 = ∇ × 𝐮` or `ω = ∂uʸ/∂x - ∂uˣ/∂y` 
    """
    if d2:
        if isinstance(u, tuple):
            ux, uy = u
        else:
            ux, uy = split(u)
        return Dx(uy, 0) - Dx(ux, 1)
    else:
        if isinstance(u, tuple):
            u = as_vector(u)
        return curl(u)
    

def streamfunction_from_vorticity(
    psi: Function,
    omega: Function,
    bcs: BoundaryConditions | None = None,
    neg: bool = False,
) -> list[Form]:
    """
    `∇²ψ = ±ω` depending on sign convention
    for an incompressible flow `∇·𝐮 = 0`.
    """
    rhs = omega
    if neg:
        rhs *= -1
    return poisson(psi, rhs, bcs=bcs)


def streamfunction_from_velocity(
    psi: Function,
    u: Function,
    bcs: BoundaryConditions | None = None,
    neg: bool = False,
) -> list[Form]:
    """
    `∇²ψ = ±(∂uʸ/∂x - ∂uˣ/∂y)` depending on sign convention
    for an incompressible flow `∇·𝐮 = 0`.
    """
    rhs = vorticity_from_velocity(u, d2=True)
    if neg:
        rhs *= -1
    return poisson(psi, rhs, bcs=bcs)