from typing import Callable

from ufl import (Form, Argument, FacetNormal, CellDiameter,
    TestFunction, TrialFunction, TestFunctions, TrialFunctions,
    inner, grad, div, avg, jump, Dx, Measure
)
from ufl.core.expr import Expr

from lucifex.solver import BoundaryConditions, BlockedForm
from lucifex.fem import Function, Constant
from lucifex.utils.fenicsx_utils import is_zero


def stokes_incompressible(
    up: Function,
    deviatoric_stress: Callable[[Function | Argument], Expr],
    f: Function | Constant | None = None,
    bcs: BoundaryConditions | None = None,
    blocked: bool = False,
) -> list[Form] | BlockedForm:
    """
    `∇·𝐮 = 0` \\
    `𝟎 = -∇p + ∇·𝜏(𝐮) + 𝐟`
    """
    dx = Measure('dx', up.function_space.mesh)
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
        return BlockedForm(
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

    F_dx = div(grad(v)) * div(grad(psi_trial)) * dx

    F_dS = (alpha / avg(h)) * inner(jump(grad(v), n), jump(grad(psi_trial), n)) * dS
    F_dS -= inner(jump(grad(v), n), avg(div(grad(psi_trial)))) * dS
    F_dS -= inner(avg(div(grad(v))), jump(grad(psi_trial), n)) * dS

    forms = [F_dx, F_dS]

    if not is_zero(fx):
        F_fx = v * Dx(fx, 1) * dx
        forms.append(F_fx)
    if not is_zero(fy):
        F_fy = -v * Dx(fy, 0) * dx
        forms.append(F_fy)

    return forms