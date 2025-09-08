from typing import Literal, Callable, overload
from functools import singledispatch

import numpy as np
from ufl import max_value, sqrt, inner
from ufl.core.expr import Expr
from dolfinx.fem import Function
from ufl.geometry import GeometricCellQuantity

from ..fem import LUCiFExConstant
from ..utils import dofs, cell_size_quantity, MultipleDispatchTypeError, extract_mesh


@overload
def cfl_timestep(
    u: Function | Expr,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"],
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tCFL = C minₓ(h(x) / |u(x)|)`
    """
    ...


@overload
def cfl_timestep(
    u: Function | Expr,
    h: float,
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tCFL = C h / maxₓ|u(x)|)`
    """
    ...


@overload
def cfl_timestep(
    u: float | LUCiFExConstant,
    h: float,
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
)-> float:
    """
    `∆tCFL = C h / u`
    """
    ...


@overload
def cfl_timestep(
    u: LUCiFExConstant,
    h:  GeometricCellQuantity | Literal["hmin", "hmax", "hdiam"],
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
) -> float:
    """
    `∆tCFL = C minₓ(h(x)) / u`
    """
    ...


def cfl_timestep(u, h, courant=1.0, dt_max=np.inf, dt_min=0.0, tol=1e-10):
    _lambda = _cfl_lambda_capture(u, h, tol)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _cfl_lambda_capture(velocity) -> Callable[[], float]:
    raise MultipleDispatchTypeError(velocity, _cfl_lambda_capture)


@_cfl_lambda_capture.register(Function)
@_cfl_lambda_capture.register(Expr)
def _(velocity, h, tol):
    if isinstance(velocity, Function):
        mesh = velocity.function_space.mesh
    else:
        mesh = extract_mesh(velocity)
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    return lambda: np.min(dofs(h / max_value(sqrt(inner(velocity, velocity)), tol), (mesh, 'DP', 0), use_cache=True))


@_cfl_lambda_capture.register(LUCiFExConstant)
def _(velocity: LUCiFExConstant, h, tol):
    mesh = velocity.mesh
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)
    if isinstance(h, GeometricCellQuantity):
        h = np.min(dofs(h, (mesh, 'DP', 0), use_cache=True))
    if velocity.ufl_shape == ():
        return lambda: h / max(velocity.value, tol)
    else:
        return lambda: h / max(np.linalg.norm(velocity.value, ord=2), tol)


@_cfl_lambda_capture.register(float)
@_cfl_lambda_capture.register(int)
def _(velocity, h, tol):
    if not isinstance(h, (float, int)):
        raise TypeError(f'`h` must be a `float` if `velocity` is a `float`')
    return lambda: h / max(velocity, tol)


@overload
def kinetic_timestep(
    r: Function | Expr | LUCiFExConstant | float,
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
): 
    """
    `∆tK = C / maxₓr(x) `
    """
    ...

@overload
def kinetic_timestep(
    r: LUCiFExConstant | float,
    courant: float = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
): 
    """
    `∆tK = C / r `
    """
    ...


def kinetic_timestep(r, courant = 1.0, dt_max = np.inf, dt_min = 0.0, tol = 1e-10): 
    _lambda = _kinetic_lambda_capture(r, tol)
    return _bounded_timestep(_lambda, dt_max, dt_min, courant)


@singledispatch
def _kinetic_lambda_capture(reaction) -> Callable[[], float]:
    raise MultipleDispatchTypeError(reaction)


@_kinetic_lambda_capture.register(Function)
@_kinetic_lambda_capture.register(Expr)
def _(reaction, tol):
    if isinstance(reaction, Function):
        mesh = reaction.function_space.mesh
    else:
        mesh = extract_mesh(reaction)
    return lambda: np.min(dofs(1.0 / max_value(reaction, tol), (mesh, 'DP', 0), use_cache=True))


@_kinetic_lambda_capture.register(LUCiFExConstant)
def _(reaction: LUCiFExConstant, tol):
    if reaction.ufl_shape == ():
        return lambda: 1.0 / max(reaction.value, tol)
    else:
        raise ValueError('Expected a scalar reaction')


@_kinetic_lambda_capture.register(float)
@_kinetic_lambda_capture.register(int)
def _(reaction, tol):
    return lambda: 1.0 / max(reaction, tol)


def cflk_timestep(
    u: Function | Expr | LUCiFExConstant | float,
    r: Function | Expr | LUCiFExConstant | float,
    h:  float | Literal["hmin", "hmax", "hdiam"] | GeometricCellQuantity,
    cfl_courant: float | None = 1.0,
    k_courant: float | None = 1.0,
    dt_max: float = np.inf,
    dt_min: float = 0.0,
    tol: float = 1e-10,
):
    """
    Advection-diffusion-reaction timestep combining CFL and kinetic conditions.

    `∆tCFLK = min{cK minₓ(1 / r(x)), cCFL minₓ(h(x) / |u(x)|)}` 
    """
    if cfl_courant is None:
        return kinetic_timestep(r, k_courant, dt_max, dt_min, tol)
    if k_courant is None:
        return cfl_timestep(u, h, cfl_courant, dt_max, dt_min, tol)
    _lambda_cfl = _cfl_lambda_capture(u, h, tol)
    _lambda_k = _kinetic_lambda_capture(r, tol)
    _lambda_cflk = lambda: min(
        k_courant * _lambda_k(), cfl_courant * _lambda_cfl()
    )
    return _bounded_timestep(_lambda_cflk, dt_max, dt_min, 1.0)


def _bounded_timestep(
    _lambda: Callable[[], float],
    dt_max: float,
    dt_min: float,  
    courant: float,
) -> float:
    if np.isclose(dt_min, dt_max):
        return dt_min
    return min(dt_max, max(dt_min, courant * _lambda()))


# @copy_callable(ExpressionSolver[ConstantSeries, StaticConstant].from_function(cfl_timestep))
# def cfl_timestep_solver():
#     pass


# @copy_callable(ExpressionSolver[ConstantSeries, StaticConstant].from_function(kinetic_timestep))
# def kinetic_timestep_solver():
#     pass


# @copy_callable(ExpressionSolver[ConstantSeries, StaticConstant].from_function(cflk_timestep))
# def cflk_timestep_solver():
#     pass