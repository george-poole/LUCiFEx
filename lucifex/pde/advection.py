from typing import Callable

from ufl.core.expr import Expr
from ufl import (
    Form, Measure, TestFunction, TestFunction, FacetNormal,
)

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries, FiniteDifference,
    FiniteDifferenceDerivative, FiniteDifferenceArgwise, 
    DT, BE, AB1, Series,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions


from .dg_transport import dg_advection_forms
from .cg_transport import derivative_form, advection_forms, reaction_forms


def advection(
    u: FunctionSeries,
    dt: Constant,
    a: Expr | Function | Constant,
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    D_dt: FiniteDifferenceDerivative = DT,
    bcs: BoundaryConditions | None = None,
) -> tuple[Form, Form]:
    """
    `∂u/∂t + 𝐚·∇u = 0`
    """
    return advection_reaction(
        u, dt, a, 
        D_adv=D_adv, 
        D_dt=D_dt,
        bcs=bcs,
    )


def advection_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: Expr | Function | Constant,
    r: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    j: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_dt: FiniteDifferenceDerivative = DT,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = Ru + J`
    """
    v = TestFunction(u.function_space)
    dx = Measure('dx', u.function_space.mesh)
    return [
        derivative_form(v, u, dt, D_dt, dx),
        *advection_forms(v, u, a, D_adv, bcs, dx, by_parts=(bcs is not None)),
        *reaction_forms(-v, u, r, j, D_reac, D_src, dx)
    ]


def dg_advection(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    D_adv: FiniteDifference,
    D_dt: FiniteDifferenceDerivative = DT,
    bcs: BoundaryConditions | None = None,
    dx_opt: int = 0,
    dS_opt: int= 0,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = 0`
    """
    return dg_advection_reaction(
        u, dt, a,
        D_adv=D_adv, D_dt=D_dt, bcs=bcs, 
        dx_opt=dx_opt, dS_opt=dS_opt,
    )


def dg_advection_reaction(
    u: FunctionSeries,
    dt: Constant,
    a: Function | Constant,
    r: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    j: Function | Expr | Series | tuple[Callable, tuple] | None = None,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = AB1 @ BE,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_dt: FiniteDifferenceDerivative = DT,
    bcs: BoundaryConditions | None = None,
    dx_opt: int = 0,
    dS_opt: int= 0,
) -> list[Form]:
    """
    `∂u/∂t + 𝐚·∇u = Ru + J`
    """
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)
    dx = Measure('dx', u.function_space.mesh)
    dS = Measure('dS', u.function_space.mesh)
    return [
        derivative_form(v, u, dt, D_dt, dx),
        *dg_advection_forms(v, u, a, n, bcs, D_adv, dx, dS, dx_opt=dx_opt, dS_opt=dS_opt),
        *reaction_forms(-v, u, r, j, D_reac, D_src, dx),
    ]
