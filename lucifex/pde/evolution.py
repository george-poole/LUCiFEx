from typing import Callable

from ufl import Measure, TestFunction, Form
from ufl.core.expr import Expr

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    DT, AB1, FiniteDifference, FunctionSeries, ConstantSeries, Series, 
    FiniteDifferenceArgwise, ImplicitDiscretizationError
)


def evolution(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    r: Function | Expr | Series | tuple[Callable, tuple],
    D_rhs: FiniteDifference | FiniteDifferenceArgwise,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
) -> tuple[Form, Form]:
    """
    `𝜑∂u/∂t = R`
    """
    phi = D_phi(phi)
    dx = Measure('dx', u.function_space.mesh)
    v = TestFunction(u.function_space)
    F_dsdt = v * DT(u, dt) * dx
    r = D_rhs(r, trial=u)
    F_reac = -v * (1/phi) * r * dx
    return F_dsdt, F_reac



def evolution_update(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    r: Series | Expr | Function,
    D_rhs: FiniteDifference | FiniteDifferenceArgwise,
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    explicit: int | None = None,
) -> Expr:
    """
    `𝜑∂u/∂t = R`

    rearranged after finite difference discretization into the algebraic expression

    `uⁿ⁺¹ = uⁿ + (1/𝜑)Δtⁿ 𝒟(R)`

    under the assumption that 𝒟(R) with respect to `u`.
    """
    if isinstance(dt, ConstantSeries):
        dt = dt[0]
    phi = D_phi(phi)

    if isinstance(D_rhs, FiniteDifference):
        if D_rhs.is_implicit:
            raise ImplicitDiscretizationError(D_rhs, f"Reaction must be explicit in '{u.name}'.")
    else:
        if explicit is None:
            raise ValueError(f"Need to provide argument index of '{u.name}'")
        if D_rhs.finite_differences[explicit].is_implicit:
            raise ImplicitDiscretizationError(D_rhs.finite_differences[explicit], f"Reaction must be explicit in '{u.name}'.")
    
    return u[0] + (1 / phi) * dt * D_rhs(r, trial=u)