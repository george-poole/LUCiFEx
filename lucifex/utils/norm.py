from typing import Literal

import ufl
import numpy as np
from ufl.core.expr import Expr
from dolfinx.fem import Function

def l_norm(
    u:  Function,
    p: float | Literal['inf'],
) -> float:
    """
    `‖u‖ℓₚ = ( Σᵢ|Uᵢ|ᵖ )¹ᐟᵖ`
    """
    if p == "inf":
        p = np.inf
    return np.linalg.norm(u.x.array[:], p)


def L_norm(
    u:  Function | Expr,
    p: float,
) -> Expr:
    """
    Integrand for norm `‖u‖ℒₚ = ∫ |u(x)|ᵖ dx`
    """
    return abs(u) ** p


def div_norm(
    u:  Function | Expr,
    p: float,
) -> Expr:
    """
    Integrand for norm `‖u‖divₚ = ∫ |∇·u(x)|ᵖ dx = ‖∇·u‖ℒₚ`
    """
    return L_norm(ufl.div(u), p)