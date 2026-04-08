from typing import Literal, Callable

import numpy as np
from ufl.core.expr import Expr
from ufl import inner, div, grad, SpatialCoordinate
from dolfinx.mesh import Mesh
from dolfinx.fem import Function, FunctionSpace

from ..utils.fenicsx_utils import dofs, mesh_integral, set_function_interpolate, set_function


def l_norm(
    u: Function,
    p: float | Literal['inf'],
) -> float:
    """
    `‖u‖ℓₚ = ( Σᵢ|Uᵢ|ᵖ )¹ᐟᵖ`

    where `{Uᵢ}` are the degrees of freedom.
    """
    if p == "inf":
        p = np.inf
    return np.linalg.norm(u.x.array[:], p)


@mesh_integral
def L_norm(
    u:  Function | Expr,
    p: float,
    absolute: bool = True,
) -> Expr:
    """
    `‖u‖ℒₚᵖ = ∫ |u(𝐱)|ᵖ dx`

    Note that raising the evaluated integral to the power of `1/p` 
    is required to recover the usual definition of the divergence norm.
    """
    if absolute:
        _abs = abs
    else:
        _abs = lambda u: u
    return _abs(u) ** p


@mesh_integral
def div_norm(
    u:  Function | Expr,
    p: float,
    absolute: bool = True,
) -> Expr:
    """
    `‖𝐮‖divₚᵖ = ∫ |∇·𝐮(𝐱)|ᵖ dx = ‖∇·𝐮‖ℒₚᵖ`

    Note that raising the evaluated integral to the power of `1/p` 
    is required to recover the usual definition of the divergence norm.
    """
    return L_norm(div(u), p, absolute)


@mesh_integral
def grad_norm(
    u:  Function | Expr,
    p: float,
    absolute: bool = True,
) -> Expr:
    """
    `‖u‖gradₚᵖ = ∫ |∇u(𝐱)·∇u(𝐱)|ᵖ dx = ‖∇u·∇u‖ℒₚ`

    Note that raising the evaluated integral to the power of `1/p` 
    is required to recover the usual definition of the divergence norm.
    """
    return L_norm(inner(grad(u), grad(u)), p, absolute)


def extrema(
    u: Function | Expr,
    elem: tuple[str, int] | tuple[Mesh, str, int] | FunctionSpace | None = ('P', 1),
) -> tuple[float, float]:
    """
    `minₓ(u(𝐱)), maxₓ(u(𝐱))` or `minₓ|𝐮(𝐱)|, maxₓ|𝐮(𝐱)|`
    """
    _dofs = dofs(u, elem, l2_norm=True, use_cache=True, avoid_new=True) 
    return np.min(_dofs), np.max(_dofs)


def minimum(
    u: Function | Expr,
    elem: tuple[str, int] | tuple[Mesh, str, int] | FunctionSpace | None = ('P', 1),
) -> float:
    """
    `minₓ(u(𝐱))` or `minₓ|𝐮(𝐱)|`
    """
    return extrema(u, elem)[0]


def maximum(
    u: Function | Expr,
    elem: tuple[str, int] | tuple[Mesh, str, int] | FunctionSpace | None = ('P', 1),
) -> float:
    """
    `maxₓ(u(𝐱))` or `maxₓ|𝐮(𝐱)|`
    """
    return extrema(u, elem)[1]


def error_norms(
    u_numerical: Function,
    u_exact: Callable[[SpatialCoordinate], Expr],
    degree_raise: int = 0,
    norm_factory: Callable[[Function | Expr], float] | None = None,
) -> tuple[float, float, float, float]:
    """
    Computes norm of error as `Expr`, norm of error as `Function` 
    and `ℓₚ` norms with `p ∈ {2, ∞}`.
    
    Default `norm_factory` is the `ℒ₂`-norm.
    """
    fs = u_numerical.function_space
    if degree_raise:
        family = fs.ufl_element().family()
        degree = fs.ufl_element().degree()
        fs = FunctionSpace(fs.mesh, (family, degree + degree_raise))

    if norm_factory is None:
        p = 2.0
        normalize = lambda e: e ** (1/p)
        norm_factory = lambda u: L_norm('dx', normalize=normalize)(u, p)
    
    x = SpatialCoordinate(fs.mesh) # TODO or fs_old.mesh ?
    ue_expr = u_exact(x)
    u_func = Function(fs)
    set_function_interpolate(u_func, u_numerical)
    ue_func = Function(fs)
    set_function_interpolate(ue_func, ue_expr)
    e_func = Function(fs)
    set_function(e_func, ue_func.x.array - u_func.x.array, ':')

    norm_expr = norm_factory(ue_expr - u_func)
    norm_func = norm_factory(e_func)
    l2_norm = l_norm(e_func, 2.0)
    linf_norm = l_norm(e_func, 'inf')

    return norm_expr, norm_func, l2_norm, linf_norm