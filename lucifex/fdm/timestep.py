from typing import Literal, Callable, overload
from functools import singledispatch
from collections.abc import Iterable

import numpy as np
from ufl import max_value, sqrt, inner
from ufl.core.expr import Expr
from dolfinx.fem import Function
from ufl.geometry import GeometricCellQuantity

from ..fem import Constant
from ..utils import dofs, cell_size_quantity, MultipleDispatchTypeError, extract_mesh


@overload
def cfl_timestep(
    a: Function | Expr,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"],
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tCFL = c minₓ(h(x) / |𝐚(x)|)` where `c` is the Courant number.
    """
    ...


@overload
def cfl_timestep(
    a: Function | Expr,
    h: float,
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tCFL = c h / maxₓ|𝐚(x)|)` where `c` is the Courant number.
    """
    ...


@overload
def cfl_timestep(
    a: float | Constant,
    h: float,
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tCFL = c h / a` where `c` is the Courant number.
    """
    ...


@overload
def cfl_timestep(
    a: Constant,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"],
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
) -> float:
    """
    `∆tCFL = c minₓ(h(x)) / u`
    """
    ...


def cfl_timestep(a, h, courant=1.0, dt_max=np.inf, dt_min=0.0, tol=1e-10):
    _lambda = _cfl_dt_evaluation(a, h, tol)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _cfl_dt_evaluation(velocity, *_, **__) -> Callable[[], float]:
    raise MultipleDispatchTypeError(velocity, _cfl_dt_evaluation)


@_cfl_dt_evaluation.register(Function)
@_cfl_dt_evaluation.register(Expr)
def _(velocity, h, tol):
    if isinstance(velocity, Function):
        mesh = velocity.function_space.mesh
    else:
        mesh = extract_mesh(velocity)
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    return lambda: np.min(dofs(h / max_value(sqrt(inner(velocity, velocity)), tol), (mesh, 'DP', 0), use_cache=True))


@_cfl_dt_evaluation.register(Constant)
def _(velocity: Constant, h, tol):
    mesh = velocity.mesh
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    if isinstance(h, GeometricCellQuantity):
        h = np.min(dofs(h, (mesh, 'DP', 0), use_cache=True))
    if velocity.ufl_shape == ():
        return lambda: h / max(velocity.value, tol)
    else:
        return lambda: h / max(np.linalg.norm(velocity.value, ord=2), tol)


@_cfl_dt_evaluation.register(float)
@_cfl_dt_evaluation.register(int)
def _(velocity, h, tol):
    if not isinstance(h, (float, int)):
        raise TypeError(f'`h` must be a `float` if `velocity` is a `float`')
    return lambda: h / max(velocity, tol)


@_cfl_dt_evaluation.register(Iterable)
def _(velocity: Iterable[float], h, tol):
    norm = np.linalg.norm(velocity, 2)
    return _cfl_dt_evaluation(norm, h, tol)


@overload
def reactive_timestep(
    r: Function | Expr | Constant | float,
    courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
): 
    """
    `∆tR = c / maxₓR(x)` where `c` is the Courant number.
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
    `∆tR = c / R` where `c` is the Courant number.
    """
    ...


def reactive_timestep(r, courant = 1.0, dt_max = np.inf, dt_min = 0.0, tol = 1e-10): 
    _lambda = _reactive_dt_evaluation(r, tol)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _reactive_dt_evaluation(reaction, *_, **__) -> Callable[[], float]:
    raise MultipleDispatchTypeError(reaction)


@_reactive_dt_evaluation.register(Function)
@_reactive_dt_evaluation.register(Expr)
def _(reaction, tol):
    if isinstance(reaction, Function):
        mesh = reaction.function_space.mesh
    else:
        mesh = extract_mesh(reaction)
    return lambda: np.min(dofs(1.0 / max_value(reaction, tol), (mesh, 'DP', 0), use_cache=True))


@_reactive_dt_evaluation.register(Constant)
def _(reaction: Constant, tol):
    if reaction.ufl_shape == ():
        return lambda: 1.0 / max(reaction.value, tol)
    else:
        raise ValueError('Expected a scalar reaction')


@_reactive_dt_evaluation.register(float)
@_reactive_dt_evaluation.register(int)
def _(reaction, tol):
    return lambda: 1.0 / max(reaction, tol)


def diffusive_timestep(
    d: Function | Expr | Constant | float,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"] | float,
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
):
    """
    `∆tD = c minₓ(h²(x) / D(x))` where `c` is the Courant number.
    """
    _lambda = _diffusive_dt_evaluation(d, h, tol)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _diffusive_dt_evaluation(diffusion, *_, **__) -> Callable[[], float]:
    raise MultipleDispatchTypeError(diffusion, _diffusive_dt_evaluation)


@_diffusive_dt_evaluation.register(Function)
@_diffusive_dt_evaluation.register(Expr)
def _(diffusion, h, tol):
    if isinstance(diffusion, Function):
        mesh = diffusion.function_space.mesh
    else:
        mesh = extract_mesh(diffusion)
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    return lambda: np.min(dofs(h**2 / max_value(diffusion, tol), (mesh, 'DP', 0), use_cache=True))


@_diffusive_dt_evaluation.register(Constant)
def _(diffusion: Constant, h, tol):
    mesh = diffusion.mesh
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    if isinstance(h, GeometricCellQuantity):
        h = np.min(dofs(h, (mesh, 'DP', 0), use_cache=True))
    if diffusion.ufl_shape == ():
        return lambda: h**2 / max(diffusion.value, tol)
    else:
        return lambda: h**2 / max(np.max(diffusion.value), tol)


@_diffusive_dt_evaluation.register(float)
@_diffusive_dt_evaluation.register(int)
def _(d, h, tol):
    if not isinstance(h, (float, int)):
        raise TypeError(f'`h` must be a `float` if `velocity` is a `float`')
    return lambda: h**2 / max(d, tol)


def diffusive_reactive_timestep(
    d: Function | Expr | Constant | float,
    r: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    d_courant: float | None = 1.0,
    r_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
):
    if d_courant is None and r_courant is not None:
        return reactive_timestep(r, r_courant, dt_max, dt_min, tol)
    if r_courant is None and d_courant is not None:
        return diffusive_timestep(d, h, d_courant, dt_max, dt_min, tol)
    if d_courant is None and r_courant is None:
        return dt_max
    
    _lambda_d = _diffusive_dt_evaluation(h, d, tol)
    _lambda_r = _reactive_dt_evaluation(r, tol)
    _lambda_dr = lambda: min(
        d_courant * _lambda_d(), 
        r_courant * _lambda_r(), 
    )
    return _bounded_timestep(_lambda_dr, dt_max, dt_min, 1.0)


def cflr_timestep(
    a: Function | Expr | Constant | float,
    r: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    cfl_courant: float | None = 1.0,
    r_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
):
    """
    Calculates a timestep combining CFL and reactive constraints.

    `∆tCFLR = min{cCFL minₓ(h(x) / |𝐚(x)|), cR minₓ(1 / R(x))}` 
    """
    if cfl_courant is None and r_courant is not None:
        return reactive_timestep(r, r_courant, dt_max, dt_min, tol)
    if r_courant is None and cfl_courant is not None:
        return cfl_timestep(a, h, cfl_courant, dt_max, dt_min, tol)
    if cfl_courant is None and r_courant is None:
        return dt_max

    _lambda_cfl = _cfl_dt_evaluation(a, h, tol)
    _lambda_r = _reactive_dt_evaluation(r, tol)
    _lambda_cflr = lambda: min(
        r_courant * _lambda_r(), cfl_courant * _lambda_cfl()
    )
    return _bounded_timestep(_lambda_cflr, dt_max, dt_min, 1.0)


def cfld_timestep(
    a: Function | Expr | Constant | float,
    d: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    cfl_courant: float | None = 1.0,
    d_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
):
    """
    Calculates a timestep combining CFL and diffusive constraints.

    `∆tCFLD = min{cCFL minₓ(h(x) / |𝐚(x)|), cD minₓ(h²(x) / D(x))}` 
    """
    if cfl_courant is None and d_courant is not None:
        return diffusive_timestep(d, h, d_courant, dt_max, dt_min, tol)
    if d_courant is None and cfl_courant is not None:
        return cfl_timestep(a, h, cfl_courant, dt_max, dt_min, tol)
    if cfl_courant is None and d_courant is None:
        return dt_max

    _lambda_cfl = _cfl_dt_evaluation(a, h, tol)
    _lambda_d = _diffusive_dt_evaluation(h, d, tol)
    _lambda_cfld = lambda: min(
        cfl_courant * _lambda_cfl(),
        d_courant * _lambda_d(), 
    )
    return _bounded_timestep(_lambda_cfld, dt_max, dt_min, 1.0)


def cfldr_timestep(
    a: Function | Expr | Constant | float,
    d: Function | Expr | Constant | float,
    r: Function | Expr | Constant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    cfl_courant: float | None = 1.0,
    d_courant: float | None = 1.0,
    r_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
):
    """
    Calculates a timestep combining CFL, diffusive and reactive constraints.

    `∆tCFLD = min{cCFL minₓ(h(x) / |𝐚(x)|), cD minₓ(h²(x) / D(x)), cR minₓ(1 / R(x))}` 
    """
    match cfl_courant, d_courant, r_courant:
        case None, None, None:
            return dt_max
        case _, None, None:
            return cfl_timestep(a, h, cfl_courant, dt_max, dt_min)
        case _, _, None:
            return cfld_timestep(a, d, h, cfl_courant, d_courant, dt_max, dt_min)
        case _, None, _:
            return cflr_timestep(a, r, h, cfl_courant, r_courant, dt_max, dt_min)
        case None, _, _:
            return diffusive_reactive_timestep(d, r, h, d_courant, r_courant, dt_max, dt_min)
        
    _lambda_cfl = _cfl_dt_evaluation(a, h, tol)
    _lambda_d = _diffusive_dt_evaluation(h, d, tol)
    _lambda_r = _reactive_dt_evaluation(r, tol)
    _lambda_cfldr = lambda: min(
        cfl_courant * _lambda_cfl(),
        d_courant * _lambda_d(), 
        r_courant * _lambda_r(), 
    )
    return _bounded_timestep(_lambda_cfldr, dt_max, dt_min, 1.0)
        

def _bounded_timestep(
    _lambda: Callable[[], float],
    dt_max: float,
    dt_min: float,  
    courant: float | None,
) -> float:
    if courant is None:
        return dt_max
    if np.isclose(dt_min, dt_max):
        return dt_min
    return min(dt_max, max(dt_min, courant * _lambda()))
