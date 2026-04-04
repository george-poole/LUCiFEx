from typing import Literal, Callable, overload
from functools import singledispatch
from collections.abc import Iterable

import numpy as np
from ufl import max_value, sqrt, inner, tr
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from ufl.geometry import GeometricCellQuantity

from ..fem import Constant
from ..utils.fenicsx_utils import dofs, cell_size_quantity, extract_mesh, is_tensor
from ..utils.py_utils import OverloadTypeError, LazyEvaluator


@overload
def advective_timestep(
    a: Function | Expr,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"],
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tA = cA minₓ(h(𝐱) / |𝐚(𝐱)|)` where `cA` is the advective Courant number.
    """
    ...


@overload
def advective_timestep(
    a: Function | Expr,
    h: float,
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
    mesh: Mesh | None = None,
)-> float:
    """
    `∆tA = cA h / maxₓ|𝐚(𝐱)|)` where `cA` is the advective Courant number.
    """
    ...


@overload
def advective_timestep(
    a: float | Constant,
    h: float,
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tA = cA h / a` where `cA` is the advective Courant number.
    """
    ...


@overload
def advective_timestep(
    a: Constant,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"],
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
) -> float:
    """
    `∆tA = cA minₓ(h(𝐱)) / u` where `cA` is the advective Courant number.
    """
    ...


def advective_timestep(a, h, courant=1.0, dt_max=np.inf, dt_min=0.0, tol=1e-10, mesh=None):
    _lambda = _advective_dt_evaluator(a, h, tol, mesh)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _advective_dt_evaluator(velocity, *_, **__) -> LazyEvaluator[float]:
    raise OverloadTypeError(velocity, _advective_dt_evaluator)


@_advective_dt_evaluator.register(Function)
@_advective_dt_evaluator.register(Expr)
def _(velocity, h, tol, mesh):
    if mesh is None:
        if isinstance(velocity, Function):
            mesh = velocity.function_space.mesh
        else:
            mesh = extract_mesh(velocity)
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    return lambda: np.min(
        dofs(h / max_value(sqrt(inner(velocity, velocity)), tol), (mesh, 'DP', 0), use_cache=True),
    )


@_advective_dt_evaluator.register(Constant)
def _(velocity: Constant, h, tol, mesh):
    if mesh is None:
        mesh = velocity.mesh
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    if isinstance(h, GeometricCellQuantity):
        h = np.min(dofs(h, (mesh, 'DP', 0), use_cache=True))
    if velocity.ufl_shape == ():
        return lambda: h / max(velocity.value, tol)
    else:
        return lambda: h / max(np.linalg.norm(velocity.value, ord=2), tol)


@_advective_dt_evaluator.register(float)
@_advective_dt_evaluator.register(int)
@_advective_dt_evaluator.register(np.floating)
@_advective_dt_evaluator.register(np.integer)
def _(velocity, h, tol, _):
    if not isinstance(h, (float, int)):
        raise TypeError(f'`h` must be a `float` if `velocity` is a `float`')
    return lambda: h / max(velocity, tol)


@_advective_dt_evaluator.register(Iterable)
def _(velocity: Iterable[float], h, tol, mesh):
    norm = np.linalg.norm(velocity, 2)
    return _advective_dt_evaluator(norm, h, tol, mesh)


def diffusive_timestep(
    d: Function | Expr | Constant | float,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"] | float,
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
    mesh: Mesh | None = None,
):
    """
    `∆tD = cD minₓ(h²(𝐱) / 2|D(𝐱)|)` where `cD` is the diffusive Courant number.

    If `D` is a tensor, then `|D| = tr(D) / dim(D)`.
    """
    _lambda = _diffusive_dt_evaluator(d, h, tol, mesh)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _diffusive_dt_evaluator(diffusion, *_, **__) -> LazyEvaluator[float]:
    raise OverloadTypeError(diffusion, _diffusive_dt_evaluator)


@_diffusive_dt_evaluator.register(Function)
@_diffusive_dt_evaluator.register(Expr)
def _(diffusion, h, tol, mesh):
    if mesh is None:
        if isinstance(diffusion, Function):
            mesh = diffusion.function_space.mesh
        else:
            mesh = extract_mesh(diffusion)
        if isinstance(h, str):
            h = cell_size_quantity(mesh, h)
    if is_tensor(diffusion):
        diffusion = tr(diffusion) / mesh.geometry.dim
    return lambda: np.min(
        dofs(0.5 * h**2 / max_value(diffusion, tol), (mesh, 'DP', 0), use_cache=True),
    )


@_diffusive_dt_evaluator.register(Constant)
def _(diffusion: Constant, h, tol, mesh):
    if mesh is None:
        mesh = diffusion.mesh
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    if isinstance(h, GeometricCellQuantity):
        h = np.min(dofs(h, (mesh, 'DP', 0), use_cache=True))
    if diffusion.ufl_shape == ():
        return lambda: 0.5 * h**2 / max(diffusion.value, tol)
    else:
        return lambda: 0.5 * h**2 / max(np.max(diffusion.value), tol)


@_diffusive_dt_evaluator.register(float)
@_diffusive_dt_evaluator.register(int)
@_diffusive_dt_evaluator.register(np.floating)
@_diffusive_dt_evaluator.register(np.integer)
def _(d, h, tol, _):
    if not isinstance(h, (float, int)):
        raise TypeError(f'`h` must be a `float` if `velocity` is a `float`')
    return lambda: 0.5 * h**2 / max(d, tol)


@overload
def reactive_timestep(
    r: Function | Expr | Constant | float,
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
): 
    """
    `∆tR = cR / maxₓR(𝐱)` where `cR` is the reactive Courant number.
    """
    ...

@overload
def reactive_timestep(
    r: Constant | float,
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
): 
    """
    `∆tR = cR / R` where `cR` is the reactive Courant number.
    """
    ...


def reactive_timestep(r, courant = 1.0, dt_max = np.inf, dt_min = 0.0, tol = 1e-10, mesh=None): 
    _lambda = _reactive_dt_evaluator(r, tol, mesh)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _reactive_dt_evaluator(reaction, *_, **__) -> LazyEvaluator[float]:
    raise OverloadTypeError(reaction)


@_reactive_dt_evaluator.register(Function)
@_reactive_dt_evaluator.register(Expr)
def _(reaction, tol, mesh):
    if mesh is None:
        if isinstance(reaction, Function):
            mesh = reaction.function_space.mesh
        else:
            mesh = extract_mesh(reaction)
    return lambda: np.min(
        dofs(1.0 / max_value(reaction, tol), (mesh, 'DP', 0), use_cache=True),
    )


@_reactive_dt_evaluator.register(Constant)
def _(reaction: Constant, tol, _):
    if reaction.ufl_shape == ():
        return lambda: 1.0 / max(reaction.value, tol)
    else:
        raise ValueError('Expected a scalar reaction')


@_reactive_dt_evaluator.register(float)
@_reactive_dt_evaluator.register(int)
@_reactive_dt_evaluator.register(np.floating)
@_reactive_dt_evaluator.register(np.integer)
def _(reaction, tol, _):
    return lambda: 1.0 / max(reaction, tol)


def diffusive_reactive_timestep(
    d: Function | Expr | Constant | float,
    r: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    d_courant: float | None = 1.0,
    r_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
    mesh: Mesh | None = None,
):
    if d_courant is None and r_courant is not None:
        return reactive_timestep(r, r_courant, dt_max, dt_min, tol, mesh)
    if r_courant is None and d_courant is not None:
        return diffusive_timestep(d, h, d_courant, dt_max, dt_min, tol, mesh)
    if d_courant is None and r_courant is None:
        return dt_max
    
    _lambda_d = _diffusive_dt_evaluator(d, h, tol, mesh)
    _lambda_r = _reactive_dt_evaluator(r, tol, mesh)
    _lambda_dr = lambda: min(
        d_courant * _lambda_d(), 
        r_courant * _lambda_r(), 
    )
    return _bounded_timestep(_lambda_dr, dt_max, dt_min, 1.0)


def advective_diffusive_timestep(
    a: Function | Expr | Constant | float,
    d: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    a_courant: float | None = 1.0,
    d_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
    mesh: Mesh | None = None,
):
    """
    Calculates a timestep combining advective and diffusive constraints.

    `∆tAD = min{cA minₓ(h(𝐱) / |𝐚(𝐱)|), cD minₓ(h²(𝐱) / |D(𝐱)|)}` 
    """
    if a_courant is None and d_courant is not None:
        return diffusive_timestep(d, h, d_courant, dt_max, dt_min, tol, mesh)
    if d_courant is None and a_courant is not None:
        return advective_timestep(a, h, a_courant, dt_max, dt_min, tol, mesh)
    if a_courant is None and d_courant is None:
        return dt_max

    _lambda_a = _advective_dt_evaluator(a, h, tol, mesh)
    _lambda_d = _diffusive_dt_evaluator(d, h, tol, mesh)
    _lambda_ad = lambda: min(
        a_courant * _lambda_a(),
        d_courant * _lambda_d(), 
    )
    return _bounded_timestep(_lambda_ad, dt_max, dt_min, 1.0)


def advective_reactive_timestep(
    a: Function | Expr | Constant | float,
    r: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    a_courant: float | None = 1.0,
    r_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
    mesh: Mesh | None = None,
):
    """
    Calculates a timestep combining advective and reactive constraints.

    `∆tAR = min{cA minₓ(h(𝐱) / |𝐚(𝐱)|), cR minₓ(1 / R(𝐱))}` 
    """
    if a_courant is None and r_courant is not None:
        return reactive_timestep(r, r_courant, dt_max, dt_min, tol, mesh)
    if r_courant is None and a_courant is not None:
        return advective_timestep(a, h, a_courant, dt_max, dt_min, tol, mesh)
    if a_courant is None and r_courant is None:
        return dt_max

    _lambda_a = _advective_dt_evaluator(a, h, tol, mesh)
    _lambda_r = _reactive_dt_evaluator(r, tol, mesh)
    _lambda_ar = lambda: min(
        r_courant * _lambda_r(), a_courant * _lambda_a()
    )
    return _bounded_timestep(_lambda_ar, dt_max, dt_min, 1.0)


def adr_timestep(
    a: Function | Expr | Constant | float,
    d: Function | Expr | Constant | float,
    r: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    a_courant: float | None = 1.0,
    d_courant: float | None = 1.0,
    r_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
    mesh: Mesh | None = None,
):
    """
    Calculates a timestep combining advective, diffusive and reactive constraints.

    `∆tADR = min{cA minₓ(h(𝐱) / |𝐚(𝐱)|), cD minₓ(h²(𝐱) / D(𝐱)), cR minₓ(1 / R(𝐱))}` 
    """
    match a_courant, d_courant, r_courant:
        case None, None, None:
            return dt_max
        case _, None, None:
            return advective_timestep(a, h, a_courant, dt_max, dt_min, tol, mesh)
        case _, _, None:
            return advective_diffusive_timestep(a, d, h, a_courant, d_courant, dt_max, dt_min, tol, mesh)
        case _, None, _:
            return advective_reactive_timestep(a, r, h, a_courant, r_courant, dt_max, dt_min, tol, mesh)
        case None, _, _:
            return diffusive_reactive_timestep(d, r, h, d_courant, r_courant, dt_max, dt_min, tol, mesh)
        
    _lambda_a = _advective_dt_evaluator(a, h, tol, mesh)
    _lambda_d = _diffusive_dt_evaluator(d, h, tol, mesh)
    _lambda_r = _reactive_dt_evaluator(r, tol, mesh)
    _lambda_adr = lambda: min(
        a_courant * _lambda_a(),
        d_courant * _lambda_d(), 
        r_courant * _lambda_r(), 
    )
    return _bounded_timestep(_lambda_adr, dt_max, dt_min, 1.0)
        

def _bounded_timestep(
    evaluator: LazyEvaluator[float],
    dt_max: float,
    dt_min: float,  
    courant: float | None,
) -> float:
    if courant is None:
        return dt_max
    if np.isclose(dt_min, dt_max):
        return dt_min
    return min(dt_max, max(dt_min, courant * evaluator()))


@overload
def peclet(h: float, a: float, d: float) -> Expr | float:
    """
    Peclet number `Pe = |𝐚|h / 2D`
    """
    ...


@overload
def peclet(h: Expr | float, a: Expr | float, d: Expr | float) -> Expr:
    """
    Local Peclet number `Pe(x) = |𝐚(x)|h(x) / 2D(x)`
    """
    ...


def peclet(h, a, d):
    return 0.5 * a * h / d


@overload
def peclet_argument(
    Pe, *, h, a
):
    """
    `D = |𝐚|h / 2Pe`
    """
    ...


@overload
def peclet_argument(
    Pe, *, h, d
):
    """
    |𝐚| = 2Pe D / h`
    """
    ...


@overload
def peclet_argument(
    Pe, *, a, d
):
    """
    h = 2Pe D / |𝐚|`
    """
    ...


def peclet_argument(
    Pe,
    *,
    h=None,
    a=None,
    d=None,
):
    match h, a, d:
        case _, _, None:
            return 0.5 * a * h / Pe
        case _, None, _:
            return 2 * Pe * d / h
        case None, _, _:
            return 2 * Pe * d / a
        case _:
            raise TypeError('Provide keyword arguments for two of `h, a, d`')


def courant_number(
    dt: float, 
    dtC: float,
) -> float:
    """
    `c = ∆t / ∆tC`
    """
    return dt / dtC