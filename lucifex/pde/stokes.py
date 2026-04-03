from typing import Callable

from ufl import (Form, Argument, FacetNormal, CellDiameter,
    TestFunction, TrialFunction, TestFunctions, TrialFunctions,
    inner, grad, div, avg, jump, Dx, Measure
)
from ufl.core.expr import Expr

from lucifex.solver import BoundaryConditions
from lucifex.fem import Function, Constant
from lucifex.fdm import FunctionSeries
from lucifex.utils.fenicsx_utils import is_none, BlockForm, extract_subspaces


def stokes_incompressible(
    up: Function | FunctionSeries,
    deviatoric_stress: Callable[[Function | Argument], Expr],
    f: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    *,
    blocked: bool = False,
    # add_zero: tuple[bool | None, bool | None] = (False, False),  # TODO
) -> list[Form] | BlockForm:
    """
    `∇·𝐮 = 0` \\
    `𝟎 = -∇p + ∇·𝜏(𝐮) + 𝐟`
    """
    dx = Measure('dx', up.function_space.mesh)
    if blocked:
        subspaces = up.function_subspaces
        v, q = (TestFunction(i) for i in subspaces)
        u, p = (TrialFunction(i) for i in subspaces)
    else:
        v, q = TestFunctions(up.function_space)
        u, p = TrialFunctions(up.function_space) 

    tau = deviatoric_stress(u)

    F_div = q * div(u) * dx
    F_pressure = -p * div(v) * dx
    F_stress = inner(grad(v), tau) * dx 

    if f is None:
        f = Constant(up.function_space.mesh, 0.0, shape=u.ufl_shape)
    F_force = -inner(v, f) * dx

    F_bcs = 0
    if bcs is not None:
        ds, tau_natural =  bcs.boundary_data(up, 'natural')
        F_bcs = sum([-inner(v, tauN) * ds(i) for i, tauN in tau_natural])

    if blocked:
        return BlockForm(
            [F_stress + F_force + F_bcs, F_pressure],
            [F_div, None],
        )
    else:
        forms = [F_div, F_pressure, F_stress, F_force]
        if F_bcs:
            forms.append(F_bcs)
        return forms


def stokes_streamfunction(
    psi: Function,
    alpha: Constant,
    fx: Function | None = None,
    fy: Function | None = None,
    mu: Function | Constant | float = 1,
    # bcs: BoundaryConditions | None = None # TODO
) -> list[Form]:
    """
    `∇⁴ψ = ∂fʸ/∂x - ∂fˣ/∂y`
    """
    dx = Measure('dx', psi.function_space.mesh)
    dS = Measure('dS', psi.function_space.mesh)
    v = TestFunction(psi.function_space)
    psi_trial = TrialFunction(psi.function_space)
    n = FacetNormal(psi.function_space.mesh)
    h = CellDiameter(psi.function_space.mesh)

    F_dx = mu * div(grad(v)) * div(grad(psi_trial)) * dx

    F_dS = mu * (alpha / avg(h)) * inner(jump(grad(v), n), jump(grad(psi_trial), n)) * dS
    F_dS -= mu * inner(jump(grad(v), n), avg(div(grad(psi_trial)))) * dS
    F_dS -= mu * inner(avg(div(grad(v))), jump(grad(psi_trial), n)) * dS

    forms = [F_dx, F_dS]

    if not is_none(fx):
        F_fx = v * Dx(fx, 1) * dx
        forms.append(F_fx)
    if not is_none(fy):
        F_fy = -v * Dx(fy, 0) * dx
        forms.append(F_fy)

    return forms