from typing import Literal

import ufl
import numpy as np
from ufl.core.expr import Expr
from dolfinx.fem import Function

from .measure_utils import integral


@integral
def l_norm(
    u:  Function,
    p: float | Literal['inf'],
) -> float:
    """
    `‖u‖ℓₚ = ( Σᵢ|Uᵢ|ᵖ )¹ᐟᵖ`

    where {Uᵢ} are the degrees of freedom.
    """
    if p == "inf":
        p = np.inf
    return np.linalg.norm(u.x.array[:], p)


@integral
def L_norm(
    u:  Function | Expr,
    p: float,
) -> Expr:
    """
    `‖u‖ℒₚ = ∫ |u(x)|ᵖ dx`

    Note that `‖u‖ℒₚ –> ‖u‖ℒₚ¹ᐟᵖ` is required
    to recover the convential definition of the 
    ℒₚ-norm.
    """
    return abs(u) ** p


@integral
def div_norm(
    u:  Function | Expr,
    p: float,
) -> Expr:
    """
    `‖u‖divₚ = ∫ |∇·u(x)|ᵖ dx = ‖∇·u‖ℒₚ`

    Note that `‖u‖divₚ –> ‖u‖divₚ¹ᐟᵖ` is required
    to recover the convential definition of the 
    divergence norm.
    """
    return L_norm(ufl.div(u), p)