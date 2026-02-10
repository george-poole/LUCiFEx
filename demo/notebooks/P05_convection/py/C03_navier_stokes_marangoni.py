import numpy as np
from ufl import as_vector, Dx

from lucifex.fdm import (
    BE, FE, CN, FiniteDifference, FiniteDifferenceArgwise, advective_timestep, 
    FunctionSeries, ConstantSeries, ExprSeries, finite_difference_order,
)
from lucifex.fem import Constant, SpatialPerturbation, sinusoid_noise
from lucifex.solver import (
    BoundaryConditions, ibvp, evaluation,
)
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.sim import configure_simulation
from lucifex.utils import CellType

from lucifex.pde.navier_stokes import ipcs_solvers
from lucifex.pde.constitutive import newtonian_stress
from lucifex.pde.advection_diffusion import advection_diffusion

from .C03_navier_stokes_thermosolutal import NAVIER_STOKES_CONVECTION_SCALINGS


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def navier_stokes_marangoni(
    # domain
    aspect: float,
    Nx: int,
    Ny: int,
    cell: str = CellType.QUADRILATERAL,
    # physical
    scaling: str = 'diffusive',
    Pr: float = 1.0,
    Ra: float = 1.0,
    Ma: float = 1.0,
    # initial perturbation
    zeta0: float = 0.8,
    zeta_eps: float = 0.01,
    c_ampl: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    # time step
    dt_max: float = 0.1,
    dt_min: float = 0.0,
    dt_courant: float = 0.75,
    # time discretization
    D_adv_ns: FiniteDifference = FE,
    D_visc_ns: FiniteDifference = CN,
    D_buoy_ns: FiniteDifference = FE,
    D_adv_c: FiniteDifference | FiniteDifferenceArgwise = (BE @ BE),
    D_diff_c: FiniteDifference = CN,
):
    scaling_map = NAVIER_STOKES_CONVECTION_SCALINGS[scaling](Ra, Pr)
    X = scaling_map['X']
    Lx = aspect * X
    Ly = 1.0 * X
    Lzeta = zeta0 * Ly
    Lzeta_eps = zeta_eps * Ly
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
    # time
    order = finite_difference_order(
        D_adv_ns, D_visc_ns, D_buoy_ns, D_adv_c, D_diff_c,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')
    # constants
    Di, Vi, Bu = scaling_map[Omega, 'Di', 'Vi', 'Bu']
    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')
    Ma = Constant(Omega, Ma, 'Ma')
    # flow
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    # transport
    c_ics = SpatialPerturbation(
        lambda x: np.exp(-(x[1] - Lzeta)**2 / Lzeta_eps),
        sinusoid_noise(['neumann', 'neumann'], [Lx, Ly], c_freq),
        [Lx, Ly],
        c_ampl,
    ) 
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    deviatoric_stress = lambda u: Vi * newtonian_stress(u, 1)
    rho = ExprSeries(c, 'rho')
    f = Bu * rho * as_vector([0, -1])
    # boundary conditions
    c_bcs = BoundaryConditions(
        ('neumann', dOmega.union, 0.0)
    )
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega['lower', 'left', 'right'], u_zero),
        ('dirichlet', dOmega['upper'], 0.0, 1),
    )  
    sigma_bcs = BoundaryConditions(
        ('natural', dOmega['upper'], as_vector([-Ma * Dx(c[0], 0), 0.0])),
    )
    # solvers
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], 'hmin', dt_courant, dt_max, dt_min,
    )
    ns_solvers = ipcs_solvers(
        u, p, dt[0], deviatoric_stress, D_adv_ns, D_visc_ns, D_buoy_ns, 
        f=f, 
        u_bcs=u_bcs, 
        sigma_bcs=sigma_bcs, 
        p_scale=Vi,
    )
    c_solver = ibvp(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, Di, D_adv_c, D_diff_c,
    )
    solvers = [dt_solver, *ns_solvers, c_solver]
    exprs_consts = [Pr, Ra, Di, Vi, Bu, rho]
    return solvers, t, dt, exprs_consts 
    