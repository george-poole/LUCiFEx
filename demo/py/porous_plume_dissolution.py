from typing import Callable

import numpy as np
from ufl.core.expr import Expr
from ufl import dx, Form, TestFunction

from lucifex.fdm import DT, FiniteDifference
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.mesh import MeshBoundary, rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, AB1, Series, 
    ExprSeries, finite_difference_order, cfl_timestep, apply_finite_difference,
    ExplicitDiscretizationError, apply_finite_difference,
)
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, bvp_solver, ibvp_solver, 
    eval_solver, interpolation_solver
)
from lucifex.utils import CellType
from lucifex.sim import configure_simulation

from .porous import (
    porous_advection_diffusion_reaction, 
    porous_advection_diffusion, 
    darcy_incompressible,
)


def evolution(
    s: FunctionSeries,
    dt: Constant,
    varphi: Function | Constant | float,
    epsilon: Constant,
    Da: Constant,
    r: Series | Expr | Function,
    D_reac: FiniteDifference | tuple[FiniteDifference, ...],
) -> Expr:        
    if isinstance(D_reac, FiniteDifference):
        if D_reac.is_implicit:
            raise ExplicitDiscretizationError(D_reac, 'Reaction must be explicit w.r.t. saturation')
    else:
        if D_reac[0].is_implicit:
            raise ExplicitDiscretizationError(D_reac[0], 'Reaction must be explicit w.r.t. saturation')

    r = apply_finite_difference(D_reac, r, s)
    return s[0] - dt * (epsilon * Da / varphi) * r


@configure_simulation(
    store_step=1,
    write_step=None,
)
def porous_plume_dissolution_rectangle(
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str = CellType.TRIANGLE,
    # physical
    Ra: float = 1000.0,
    Da: float = 1e3,
    Le: float = 1.0,
    epsilon: float = 1e-2,
    beta: float = 1.0,
    # inflow
    delta: float = 0.1,
    u_in: float = 1.0,
    b_in: float = 0.0,
    b0: float = 0.0,
    # time step
    dt_min: float = 0.0,
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    cfl_courant: float | None = 0.75,
    # time discretization
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = AB1,
    D_diff: FiniteDifference = AB1,
    D_reac: FiniteDifference = AB1,
    # linear algebra
    up_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    b_petsc: OptionsPETSc | None = None,
):
    # time
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(Omega, "t", order, ics=0.0)  
    dt = ConstantSeries(Omega, 'dt')

    # space
    Omega = rectangle_mesh((-Lx/2, Lx/2), Ly, Nx, Ny, cell=cell)
    dOmega = mesh_boundary(
        Omega, 
        {
            "left": lambda x: x[0] + 0.5 * Lx,
            "right": lambda x: x[0] - 0.5 * Lx,
            "inflow": lambda x: np.isclose(x[1], 0.0) & ((x[0] <= 0.5 * delta * Lx) & (x[0] >= -0.5 * delta * Lx)),
            "lower": lambda x: np.isclose(x[1], 0.0) & ((x[0] > 0.5 * delta * Lx) | (x[0] < -0.5 * delta * Lx)),
            "upper": lambda x: x[1] - Ly,
        },
    )

    # boundary conditions
    c_bcs = BoundaryConditions(
        ('neumann', dOmega['left', 'right', 'upper', 'lower'], 0.0),
        ('dirichlet', dOmega['inflow'], 0.0),
    )
    b_bcs = BoundaryConditions(
        ('neumann', dOmega['left', 'right', 'upper', 'lower'], 0.0),
        ('dirichlet', dOmega['inflow'], b_in),

    )
    u_bcs = BoundaryConditions(
        ('essential', dOmega['left', 'right', 'lower'], (0.0, 0.0), 0),
        ('essential', dOmega['upper'], (0.0, u_in * delta), 0),
        ('essential', dOmega['inflow'], (0.0, u_in), 0),
    )

    # constants 
    Ra = Constant(Omega, Ra, 'Ra')
    Da = Constant(Omega, Da, 'Da')
    Le = Constant(Omega, Le, 'Le')
    epsilon = Constant(Omega, epsilon, 'epsilon')
    beta = Constant(Omega, beta, 'beta')
    egx = 0
    egy = -1

    # flow
    u_fam = 'BDMCF' if Omega.topology.cell_name() == CellType.QUADRILATERAL else 'BDM'
    u_deg = 1
    up = FunctionSeries((Omega, [(u_fam, u_deg), ('DP', u_deg - 1)]), "up", order)
    u = up.sub(0, 'u')
    # transport
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=1.0)
    b = FunctionSeries((Omega, 'P', 1), 'b', order, ics=b0)
    # evolution
    s = FunctionSeries((Omega, 'P', 1), 's', order, ics=1.0)
    # constitutive
    varphi = Function((Omega, 'P', 1), 1, 'varphi')
    phi = ExprSeries(varphi * (1 - c), 'phi')
    k = ExprSeries(phi**2, 'k')
    rho = ExprSeries(c + beta * b, 'rho')
    r = Expr(s * (1 - c), 'r')

    # solvers
    up_petsc = OptionsPETSc("gmres", "lu") if up_petsc is None else up_petsc
    up_petsc['pc_factor_mat_solver_type'] = 'mumps'
    up_solver = bvp_solver(darcy_incompressible, u_bcs, petsc=up_petsc)(
        up, rho[0], k[0], 1, egx, egy,
    )
    dt_solver = eval_solver(dt, cfl_timestep)(
            u[0], cfl_h, cfl_courant, dt_max, dt_min,
        ) 

    c_solver = ibvp_solver(porous_advection_diffusion_reaction, bcs=c_bcs, petsc=c_petsc)(
        c, dt[0], phi, u, Ra, 1, Da, r, D_adv, D_diff, D_reac
    )
    b_solver = ibvp_solver(porous_advection_diffusion, bcs=b_bcs, petsc=b_petsc)(
        c, dt[0], phi, u, Ra, 1, D_adv, D_diff,
    )
    s_solver = interpolation_solver(s, evolution)(s, dt[0], varphi, )

    solvers = [up_solver, dt_solver, c_solver, b_solver, s_solver]
    namespace = [Ra, Da, Le, epsilon, beta, phi, k, rho, r]
    return solvers, t, dt, namespace