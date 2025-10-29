from typing import Callable

from ufl import dx, TestFunction, Form
from ufl.core.expr import Expr

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    DT, AB1, FiniteDifference, FunctionSeries, ConstantSeries, Series, 
    apply_finite_difference, ExplicitDiscretizationError
)


def evolution_forms(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    r: Function | Expr | Series | tuple[Callable, tuple],
    D_rhs: FiniteDifference | tuple[FiniteDifference, ...],
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
) -> tuple[Form, Form]:
    """
    `∂u/∂t = R`

    `𝜑∂u/∂t = R`
    """
    if isinstance(phi, Series):
        phi = D_phi(phi)
    v = TestFunction(u.function_space)
    F_dsdt = v * DT(u, dt) * dx
    r = apply_finite_difference(D_rhs, r, u)
    F_reac = -v * (1/phi) * r * dx
    return F_dsdt, F_reac



def evolution_expression(
    u: FunctionSeries,
    dt: Constant | ConstantSeries,
    r: Series | Expr | Function,
    D_rhs: FiniteDifference | tuple[FiniteDifference, ...],
    D_phi: FiniteDifference = AB1,
    phi: Series | Function | Expr | float = 1,
    tuple_index: int = 0,
) -> Expr:
    """
    `∂u/∂t = R` \\
    `𝜑∂u/∂t = R`

    rearranged after finite difference discretization into the algebraic expression

    `uⁿ⁺¹ = uⁿ + Δtⁿ 𝒟(R)` \\
    `uⁿ⁺¹ = uⁿ + (1/𝜑)Δtⁿ 𝒟(R)`

    under the assumption that 𝒟(R) with respect to `u`.
    """
    if isinstance(dt, ConstantSeries):
        dt = dt[0]
    if isinstance(phi, Series):
        phi = D_phi(phi)

    if isinstance(D_rhs, FiniteDifference):
        if D_rhs.is_implicit:
            raise ExplicitDiscretizationError(D_rhs, f'Reaction must be explicit w.r.t. {u.name}')
    else:
        if D_rhs[tuple_index].is_implicit:
            raise ExplicitDiscretizationError(D_rhs[tuple_index], f'Reaction must be explicit w.r.t. {u.name}')

    r = apply_finite_difference(D_rhs, r, u)
    return u[0] + (1 / phi) * dt * r


# def evolution_expression(
#     s: FunctionSeries,
#     dt: Constant | ConstantSeries,
#     varphi: Function | Constant | float,
#     epsilon: Constant,
#     Da: Constant,
#     r: Series | Expr | Function,
#     D_reac: FiniteDifference | tuple[FiniteDifference, ...],
# ) -> Expr:
#     """
#     `𝜑 ∂s/∂t = -ε Da R`

#     rearranged after finite difference discretization into the algebraic expression

#     `s¹ = s⁰ - Δt ε Da 𝒟(R) / 𝜑`.

#     under the assumption that 𝒟(R) is explicit in `s`.
#     """
#     if isinstance(dt, ConstantSeries):
#         dt = dt[0]
        
#     if isinstance(D_reac, FiniteDifference):
#         if D_reac.is_implicit:
#             raise ExplicitDiscretizationError(D_reac, 'Reaction must be explicit w.r.t. saturation')
#     else:
#         if D_reac[0].is_implicit:
#             raise ExplicitDiscretizationError(D_reac[0], 'Reaction must be explicit w.r.t. saturation')

#     r = apply_finite_difference(D_reac, r, s)
#     return s[0] - dt * (epsilon * Da / varphi) * r