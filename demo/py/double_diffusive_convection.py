from typing import Callable, TypeAlias
from types import EllipsisType

import numpy as np
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
from ufl import (dx, Form, FacetNormal, inner,
                 as_matrix, Dx, TrialFunction, TestFunction,
                 det, transpose,  as_matrix)

from lucifex.fdm import DT, FiniteDifference
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.mesh import MeshBoundary, rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, AB1, Series, 
    ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, bvp_solver, ibvp_solver, eval_solver, 
    ds_solver, interpolation_solver
)
from lucifex.utils import extremum
from lucifex.sim import Simulation, configure_simulation


from .navier_stokes import ipcs_1, ipcs_2, ipcs_3


def advection_diffusion(
    c: FunctionSeries,
    dt: Constant,
    u: FunctionSeries,
    Le: Constant,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    v = TestFunction(c.function_space)

    F_dcdt = v * DT(c, dt) * dx

    match D_adv:
        case D_adv_u, D_adv_c:
            adv = inner(D_adv_u(u, False), grad(D_adv_c(c)))
        case D_adv:
            adv = D_adv(inner(u, grad(c)))
    F_adv = v * adv * dx

    F_diff = (1/Le) * inner(grad(v), grad(D_diff(c))) * dx

    forms = [F_dcdt, F_adv, F_diff]

    if bcs is not None:
        ds, c_neumann = bcs.boundary_data(c.function_space, 'neumann')
        F_neumann = sum([(1 / Le) * v * cN * ds(i) for i, cN in c_neumann])
        forms.append(F_neumann)

    return forms


@configure_simulation(
    store_step=1,
    write_step=None,
)
def navier_stokes_double_diffusion(
    Lx: float,
    Ly: float,
):
    ...