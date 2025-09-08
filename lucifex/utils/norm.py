from typing import Literal

import ufl
import numpy as np
from ufl.core.expr import Expr
from dolfinx.fem import Function

def l_norm(
    u:  Function,
    p: float | Literal['inf'],
) -> float:
    """`‖u‖ℓₚ = ( Σᵢ|Uᵢ|ᵖ )¹ᐟᵖ`"""
    if p == "inf":
        p = np.inf
    return np.linalg.norm(u.x.array[:], p)


def L_norm(
    u:  Function | Expr,
    p: float,
) -> Expr:
    """`‖u‖ℒₚ = ( ∫ |u(x)|ᵖ dx )¹ᐟᵖ`"""
    return (abs(u) ** p) ** (1 / p)


def div_norm(
    u:  Function | Expr,
    p: float,
) -> Expr:
    """`‖u‖divₚ = ( ∫ |∇·u(x)|ᵖ dx )¹ᐟᵖ = ‖∇·u‖ℒₚ`"""
    return L_norm(ufl.div(u), p)


# l_norm_solver = expression_solver(l_norm)
# L_norm_solver = dx_solver(L_norm)
# div_norm_solver = dx_solver(div_norm)
