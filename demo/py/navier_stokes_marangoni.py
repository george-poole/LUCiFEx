from typing import Callable

import numpy as np
from ufl import as_vector, Dx

from lucifex.fdm import (
    BE, FE, CN, FiniteDifference, cfl_timestep, 
    FunctionSeries, ConstantSeries, ExprSeries, finite_difference_order,
)
from lucifex.fem import LUCiFExConstant as Constant
from lucifex.solver import (
    BoundaryConditions, ibvp_solver, eval_solver,
)
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, sinusoid_noise

from .navier_stokes import newtonian_stress, ipcs_solvers, advection_diffusion


@configure_simulation(
    store_step=1,
    write_step=None,
)
def navier_stokes_marangoni(
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str = CellType.QUADRILATERAL,
    # physical
    Ra: float = 1.0,
    Pr: float = 1.0,
    Ma: float = 1.0,
    # initial perturbation
    c_base: Callable[[np.ndarray], np.ndarray] = lambda x: 0 * x[0],
    noise_eps: float = 1e-6,
    noise_freq: tuple[int, int] = (8, 8),
    # time step
    dt_max: float = 0.1,
    dt_min: float = 0.0,
    cfl_courant: float = 0.75,
    # time discretization
    D_adv_ns: FiniteDifference = FE,
    D_visc_ns: FiniteDifference = CN,
    D_buoy_ns: FiniteDifference = FE,
    D_adv_c: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (BE, BE),
    D_diff_c: FiniteDifference = CN,
):
    # time
    order = finite_difference_order(
        D_adv_ns, D_visc_ns, D_buoy_ns, D_adv_c, D_diff_c,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')

    # space
    Omega = rectangle_mesh(Lx, Ly, Nx, Ny, cell)
    dOmega = mesh_boundary(
        Omega, 
        {
            "left": lambda x: x[0],
            "right": lambda x: x[0] - Lx,
            "lower": lambda x: x[1],
            "upper": lambda x: x[1] - Ly,
        },
    )
    dim = Omega.geometry.dim
    u_zero = [0.0] * dim

    # constants
    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')
    Ma = Constant(Omega, Ma, 'Ma')

    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        c_base,
        sinusoid_noise(['neumann', 'neumann'], [Lx, Ly], noise_freq),
        [Lx, Ly],
        noise_eps,
    ) 
    c_bcs = BoundaryConditions(
        ('neumann', dOmega.union, 0.0)
    )
    
    # flow and transport
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)

    rho = ExprSeries(-Ra * c, 'rho')
    f = rho * as_vector([0, -1])

    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega['lower', 'left', 'right'], u_zero),
        ('dirichlet', dOmega['upper'], 0.0, 1),
    )  

    sigma_bcs = BoundaryConditions(
        ('natural', dOmega['upper'], as_vector([-Ma * Dx(c[0], 0), 0.0])),
    )

    dt_solver = eval_solver(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )

    ns_solvers = ipcs_solvers(
        u, p, dt[0], 1/Pr, 1, newtonian_stress, D_adv_ns, D_visc_ns, D_buoy_ns, f, u_bcs, sigma_bcs=sigma_bcs,
    )

    c_solver = ibvp_solver(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, 1, D_adv_c, D_diff_c,
    )

    solvers = [dt_solver, *ns_solvers, c_solver]
    namespace = [Pr, Ra, Ma, rho]

    return solvers, t, dt, namespace 
    