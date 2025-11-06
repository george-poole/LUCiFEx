from typing import Literal

import numpy as np
from ufl.core.expr import Expr
from ufl import inner, div, grad
from dolfinx.fem import Function

from .measure_utils import integral


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
    mod: bool = True,
) -> Expr:
    """
    `‖u‖ℒₚ = ∫ |u(𝐱)|ᵖ dx`

    Note that `‖u‖ℒₚ –> ‖u‖ℒₚ¹ᐟᵖ` is required
    to recover the convential definition of the 
    ℒₚ-norm.
    """
    if mod:
        _abs = abs
    else:
        _abs = lambda u: u

    return _abs(u) ** p


@integral
def div_norm(
    u:  Function | Expr,
    p: float,
    mod: bool = True,
) -> Expr:
    """
    `‖𝐮‖divₚ = ∫ |∇·𝐮(𝐱)|ᵖ dx = ‖∇·𝐮‖ℒₚ`

    Note that `‖u‖divₚ –> ‖u‖divₚ¹ᐟᵖ` is required
    to recover the convential definition of the 
    divergence norm.
    """
    return L_norm(div(u), p, mod)


@integral
def grad_norm(
    u:  Function | Expr,
    p: float,
    mod: bool = True,
) -> Expr:
    """
    `‖u‖gradₚ = ∫ |∇u(𝐱)·∇u(𝐱)|ᵖ dx = ‖∇u·∇u‖ℒₚ`

    Note that `‖u‖divₚ –> ‖u‖divₚ¹ᐟᵖ` is required
    to recover the convential definition of the 
    divergence norm.
    """
    return L_norm(inner(grad(u), grad(u)), p, mod)