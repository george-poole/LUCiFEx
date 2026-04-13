import numpy as np

from lucifex.fdm import (
    FiniteDifference, advective_timestep, 
    FunctionSeries, ConstantSeries, finite_difference_order,
)
from lucifex.fem import Constant, Function
from lucifex.solver import (
    BoundaryConditions, OptionsJIT, evaluation, bvp,
)
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.sim import configure_simulation

from lucifex.pde.navier_stokes import navier_stokes_solvers
from lucifex.pde.constitutive import newtonian_stress
from lucifex.pde.streamfunction_vorticity import streamfunction_from_velocity


@configure_simulation(
    store_delta=1,
    write_delta=None,
    jit=OptionsJIT(Ellipsis),
)
def darcy_brinkman_circle_obstacle(
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str,
    # physical
    radius: float,
    rho: float,
    mu: float,
    p_in: float,
    # timestep
    dt_max: float,
    dt_min: float,
    dt_Cu: float,
    ns_scheme: str,
    # time discretization
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    # optional
    streamfunction: bool = False,
):
    # space
    Omega = rectangle_mesh((-Lx/2, Lx/2), (-Ly/2, Ly/2), Nx, Ny, cell)
    dOmega = mesh_boundary(
        Omega,
        {
            'upper': lambda x: x[1] - Ly/2,
            'lower': lambda x: x[1] + Ly/2,
            'left': lambda x: x[0] + Lx/2,
            'right': lambda x: x[0] - Lx/2,
        }
    )
    dim = Omega.geometry.dim
    u_zero = [0.0] * dim
    # time
    order = finite_difference_order(D_adv, D_visc)
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')
    # constants
    rho = Constant(Omega, rho, 'rho')
    mu = Constant(Omega, mu, 'mu')
    # boundary conditions
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega['upper', 'lower'], u_zero),
    )
    p_bcs = BoundaryConditions(
        ('dirichlet', dOmega['left'], p_in),
        ('dirichlet', dOmega['right'], 0.0),
    )
    # constitutive
    r = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
    inverse_permeability = lambda x: 1.0 - 1.0 * (r(x) > radius)
    kInv = Function(
        (Omega, 'DP', 0), 
        inverse_permeability,
        name='kInv',
    )
    stress = lambda u: (1/rho) * newtonian_stress(u, mu)
    # flow
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    # solvers
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], 'hmin', dt_Cu, dt_max, dt_min,
    )
    ns_solvers = navier_stokes_solvers(
        ns_scheme, u, p, dt[0], stress, D_adv, D_visc, u_bcs=u_bcs, p_bcs=p_bcs,
    )
    solvers = [dt_solver, *ns_solvers]
    # optional 
    if streamfunction:
        psi = FunctionSeries((Omega, 'P', 1), name="psi")
        psi_bcs =BoundaryConditions(
            ('dirichlet', dOmega['upper', 'lower'], 0.0),
        )
        psi_solver = bvp(streamfunction_from_velocity, psi_bcs)(psi, u[0])
        solvers.append(psi_solver)

    auxiliary = (rho, mu)
    return solvers, t, dt, auxiliary
