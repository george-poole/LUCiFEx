from typing import Literal

import numpy as np
from ufl.core.expr import Expr
from ufl import inner, div, grad
from dolfinx.fem import Function

from ..utils.dofs_utils import dofs
from ..utils.mesh_utils import mesh_integral


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


@mesh_integral
def L_norm(
    u:  Function | Expr,
    p: float,
    mod: bool = True,
) -> Expr:
    """
    `‖u‖ℒₚᵖ = ∫ |u(𝐱)|ᵖ dx`

    Note that raising the evaluated integral to the power of `1/p` 
    is required to recover the usual definition of the divergence norm.
    """
    if mod:
        _abs = abs
    else:
        _abs = lambda u: u
    return _abs(u) ** p


@mesh_integral
def div_norm(
    u:  Function | Expr,
    p: float,
    mod: bool = True,
) -> Expr:
    """
    `‖𝐮‖divₚᵖ = ∫ |∇·𝐮(𝐱)|ᵖ dx = ‖∇·𝐮‖ℒₚᵖ`

    Note that raising the evaluated integral to the power of `1/p` 
    is required to recover the usual definition of the divergence norm.
    """
    return L_norm(div(u), p, mod)


@mesh_integral
def grad_norm(
    u:  Function | Expr,
    p: float,
    mod: bool = True,
) -> Expr:
    """
    `‖u‖gradₚ = ∫ |∇u(𝐱)·∇u(𝐱)|ᵖ dx = ‖∇u·∇u‖ℒₚ`

    Note that raising the evaluated integral to the power of `1/p` 
    is required to recover the usual definition of the divergence norm.
    """
    return L_norm(inner(grad(u), grad(u)), p, mod)


def extrema(
    u: Function | Expr,
    elem: tuple[str, int] = ('P', 1),
) -> tuple[float, float]:
    _dofs = dofs(u, elem, l2_norm=True, use_cache=True) 
    return np.min(_dofs), np.max(_dofs)


def minimum(
    u: Function | Expr,
    elem: tuple[str, int] = ('P', 1),
) -> float:
    _dofs = dofs(u, elem, l2_norm=True, use_cache=True)
    return np.min(_dofs)


def maximum(
    u: Function | Expr,
    elem: tuple[str, int] = ('P', 1),
) -> float:
    _dofs = dofs(u, elem, l2_norm=True, use_cache=True)
    return np.max(_dofs)