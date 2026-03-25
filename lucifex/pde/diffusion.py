from ufl.core.expr import Expr
from ufl import Form,  Measure, TestFunction, TrialFunction

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, FiniteDifferenceDerivative, 
    FiniteDifference, FiniteDifferenceArgwise, 
    DT, AB1,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import create_zero_form

from .cg_transport import derivative_form, diffusion_forms, reaction_forms


def diffusion(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_diff: FiniteDifference, # TODO tuple
    D_dt: FiniteDifferenceDerivative = DT,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t = ∇·(D·∇u)`

    `𝒟(∂u/∂t) = ∇·(𝒟(D·∇u))`
    """
    return diffusion_reaction(
        u,
        dt,
        d,
        D_diff=D_diff,
        D_dt=D_dt,
        bcs=bcs,
    )


def diffusion_reaction(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    r: Function | Constant | None = None,
    j: Function | Constant | Expr | None = None,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_dt: FiniteDifferenceDerivative = DT,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    dx = Measure('dx', u.function_space.mesh)
    return [
        derivative_form(v, u, dt, D_dt, dx),
        *diffusion_forms(-v, u, d, D_diff, bcs, dx),
        *reaction_forms(-v, u, r, j, D_reac, D_src, dx),
    ]


def steady_diffusion_reaction(
    u: Function | FunctionSeries,
    d: Constant,
    r: Function | Constant | None = None,
    j: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    add_zero: bool | None = None,
) -> tuple[Form, Form]:
    """"
    `0 = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    u_trial = TrialFunction(u.function_space)
    dx = Measure('dx', u.function_space.mesh)

    forms = [
        *diffusion_forms(v, u_trial, d, bcs=bcs, dx=dx),
        *reaction_forms(v, u_trial, r, j, dx=dx),
    ]
    if add_zero is None:
        add_zero = j is None
    if add_zero:
        forms.append(create_zero_form(v, u.function_space.mesh, dx))
    return forms

