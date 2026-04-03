from ufl.core.expr import Expr
from ufl import (
    Form, Measure, TestFunction, TrialFunction,
    TestFunctions, TrialFunctions,
)

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    FunctionSeries,
    FiniteDifference, FiniteDifferenceArgwise, 
    FiniteDifferenceDerivative, DT, DT2, AB1,
)
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import extract_subspaces, BlockForm

from .cg_transport import second_derivative_form, diffusion_forms, reaction_forms


def wave(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_diff: FiniteDifference | FiniteDifferenceArgwise, 
    D_dt: FiniteDifferenceDerivative = DT2,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂²u/∂t² = ∇·(D·∇u)`
    """
    return wave_reaction(
        u,
        dt,
        d,
        D_diff=D_diff,
        D_dt=D_dt,
        bcs=bcs,
    )


def wave_reaction(
    u: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    r: Function | Constant | None = None,
    j: Function | Constant | Expr | None = None,
    D_diff: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_reac: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_src: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_dt: FiniteDifferenceDerivative = DT2,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∂²u/∂t² = ∇·(D·∇u) + Ru + J`
    """
    v = TestFunction(u.function_space)
    dx = Measure('dx', u.function_space.mesh)
    return [
        second_derivative_form(v, u, dt, D_dt, dx),
        *diffusion_forms(-v, u, d, D_diff, bcs, dx),
        *reaction_forms(-v, u, r, j, D_reac, D_src, dx),
    ]


def wave_mixed(
    up: FunctionSeries,
    dt: Constant | float,
    d: Function | Constant | Expr,
    D_diff: FiniteDifference | FiniteDifferenceArgwise, 
    D_p: FiniteDifference,
    D_dt: FiniteDifferenceDerivative = DT,
    bcs: BoundaryConditions | None = None,
    blocked: bool = False,
)-> list[Form] | BlockForm:
    """
    `∂u/∂t = p` \\
    `∂p/∂t = ∇·(D·∇u)`
    """
    
    dx = Measure('dx', up.function_space.mesh)
    if blocked:
        subspaces = up.function_subspaces
        v, q = (TestFunction(i) for i in subspaces)
        # u, p = up.split()
        raise NotImplementedError # TODO
    else:
        v, q = TestFunctions(up.function_space)
        u, p = up.split()

    F_dudt = v * D_dt(u, dt, trial=u, blocked=blocked) * dx
    F_p = v * D_p(p, trial=p, blocked=blocked) * dx
    F_dpdt = q * D_dt(p, dt, trial=p, blocked=blocked) * dx
    F_diff = diffusion_forms(
        q, u, d, D_diff, bcs, dx,
    )
    if blocked:
        return BlockForm(
            [
                [F_dudt, F_p], 
                [sum(F_diff) if Fdiff else None, F_dpdt],
            ],
        )
    else:
        return [
            F_dudt, F_p, F_dpdt, *F_diff
        ]
