from typing import Callable

import numpy as np

from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, FiniteDifferenceArgwise,
    AB1, CN, finite_difference_order, advective_timestep,
)
from lucifex.fem import Function, Constant
from lucifex.solver import bvp, ibvp, interpolation, evaluation, BoundaryConditions
from lucifex.utils.fenicsx_utils import CellType
from lucifex.sim import configure_simulation
from lucifex.pde.streamfunction_vorticity import velocity_from_streamfunction, streamfunction_from_vorticity
from lucifex.pde.navier_stokes import navier_stokes_vorticity


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def navier_stokes_forced(
    # domain
    Lx: float = 1.0,
    Ly: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    # physical
    fx: Callable[[np.ndarray], np.ndarray] | None = None,
    fy: Callable[[np.ndarray], np.ndarray] | None = None,
    # timestep
    dt_max: float = 0.1,
    dt_min: float = 0.0,
    dt_Cu: float = 0.75,
    # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_diff: FiniteDifference = CN,
):
    # space
    mesh = rectangle_mesh(Lx, Ly, Nx, Ny, cell)
    boundary = mesh_boundary(
        mesh, 
        {
            "left": lambda x: x[0],
            "right": lambda x: x[0] - Lx,
            "lower": lambda x: x[1],
            "upper": lambda x: x[1] - Ly,
        },
    )
    # time
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(mesh, 't', ics=0.0)
    dt = ConstantSeries(mesh, 'dt')
    # flow
    psi = FunctionSeries((mesh, 'P', 2), 'psi', order)
    omega = FunctionSeries((mesh, 'P', 1), 'omega', order, ics=0.0)
    u = FunctionSeries((mesh, 'P', 1, 2), 'u', order)
    # constants
    rho = Constant(mesh, 1.0, name='rho')
    mu = Constant(mesh, 1.0, name='mu')
    if fx is not None:
        fx = Function((mesh, 'P', 1), fx, name='fx')
    if fy is not None:
        fy = Function((mesh, 'P', 1), fy, name='fy')
    # solvers
    psi_bcs = BoundaryConditions(("dirichlet", boundary.union, 0.0))
    psi_solver = bvp(streamfunction_from_vorticity, psi_bcs)(psi, omega[0])
    u_solver = interpolation(u, velocity_from_streamfunction)(psi[0])
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], 'hmin', dt_Cu, dt_max, dt_min,
    )
    omega_bcs = BoundaryConditions(("dirichlet", boundary.union, 0.0))
    omega_solver = ibvp(navier_stokes_vorticity, bcs=omega_bcs)(
        omega, dt[0], u, rho, mu, D_adv, D_diff, fx=fx, fy=fy
    )
    solvers = [psi_solver, u_solver, dt_solver, omega_solver]
    auxiliary = [rho, mu, fy]
    return solvers, t, dt, auxiliary