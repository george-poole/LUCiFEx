from typing import Callable

import numpy as np
from ufl import as_vector, Dx

from lucifex.fdm import (
    BE, FE, CN, FiniteDifference, cfl_timestep, 
    FunctionSeries, ConstantSeries, ExprSeries, finite_difference_order,
)
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.solver import (
    BoundaryConditions, bvp_solver, ibvp_solver, eval_solver,
)
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, sinusoid_noise

from .double_diffusive_convection import advection_diffusion
from .navier_stokes import newtonian_stress, ipcs_1, ipcs_2, ipcs_3


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
    D_force_ns: FiniteDifference = FE,
    D_adv_c: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (BE, BE),
    D_diff_c: FiniteDifference = CN,
):
    order = finite_difference_order(
        D_adv_ns, D_visc_ns, D_force_ns, D_adv_c, D_diff_c,
    )
    Lx = 2.0
    Ly = 1.0
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
    zero = [0.0] * dim

    c_bcs = BoundaryConditions(
        ('neumann', dOmega.union, 0.0)
    )
    c_ics = SpatialPerturbation(
        c_base,
        sinusoid_noise(['neumann', 'neumann'], [Lx, Ly], noise_freq),
        [Lx, Ly],
        noise_eps,
    ) 

    t = ConstantSeries(Omega, 't', ics=0.0)
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    dt = ConstantSeries(Omega, 'dt')

    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')
    Ma = Constant(Omega, Ma, 'Ma')
    rho = ExprSeries(-Ra * c, 'rho')
    f = rho * as_vector([0, -1])

    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega['lower', 'left', 'right'], zero),
        ('dirichlet', dOmega['upper'], 0.0, 1),
    )  

    sigma_bcs = BoundaryConditions(
        ('natural', dOmega['upper'], as_vector([-Ma * Dx(c[0], 0), 0.0])),
    )

    dt_solver = eval_solver(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )

    ipcs1_solver = ibvp_solver(ipcs_1, bcs=u_bcs)(
        u, p, dt[0], 1/Pr, 1, newtonian_stress, D_adv_ns, D_visc_ns, D_force_ns, f, sigma_bcs,
    )
    ipcs2_solver = bvp_solver(ipcs_2, future=True)(
        p, u, dt[0], 1/Pr,
    )
    ipcs3_solver = bvp_solver(ipcs_3, future=True, overwrite=True)(
        u, p, dt[0], 1/Pr,
    )

    c_solver = ibvp_solver(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, 1, D_adv_c, D_diff_c,
    )

    solvers = [dt_solver, ipcs1_solver, ipcs2_solver, ipcs3_solver, c_solver]
    namespace = [Pr, Ra, Ma, rho]

    return solvers, t, dt, namespace 
    