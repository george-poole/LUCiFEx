from typing import Callable

from dolfinx.fem import Function, Constant
from ufl import dx, TestFunction, Form
from ufl.core.expr import Expr

from lucifex.fdm import (
    DT, FiniteDifference, FunctionSeries, ConstantSeries, Series, 
    apply_finite_difference, ExplicitDiscretizationError
)


def evolution_forms(
    s: FunctionSeries,
    dt: Constant | ConstantSeries,
    r: Function | Expr | Series | tuple[Callable, tuple],
    D_rhs: FiniteDifference | tuple[FiniteDifference, ...],
) -> tuple[Form, Form]:
    """
    `∂s/∂t = R`
    """
    v = TestFunction(s.function_space)
    F_dsdt = v * DT(s, dt) * dx
    r = apply_finite_difference(D_rhs, r, s)
    F_reac = -v * r * dx
    return F_dsdt, F_reac



def evolution_expression(
    s: FunctionSeries,
    dt: Constant | ConstantSeries,
    r: Series | Expr | Function,
    D_rhs: FiniteDifference | tuple[FiniteDifference, ...],
    tuple_index: int = 0,
) -> Expr:
    """
    `∂s/∂t = R`

    rearranged after finite difference discretization into the algebraic expression

    `sⁿ⁺¹ = sⁿ + Δtⁿ 𝒟(R)`.

    under the assumption that 𝒟(R) with respect to `s`.
    """
    if isinstance(dt, ConstantSeries):
        dt = dt[0]
        
    if isinstance(D_rhs, FiniteDifference):
        if D_rhs.is_implicit:
            raise ExplicitDiscretizationError(D_rhs, 'Reaction must be explicit w.r.t. saturation')
    else:
        if D_rhs[tuple_index].is_implicit:
            raise ExplicitDiscretizationError(D_rhs[tuple_index], 'Reaction must be explicit w.r.t. saturation')

    r = apply_finite_difference(D_rhs, r, s)
    return s[0] + dt * r


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