from ufl import as_vector, sqrt

from lucifex.fdm import FiniteDifference, FE, CN, BE
from lucifex.fem import LUCiFExConstant as Constant, LUCiFExFunction as Function
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference,
    ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.solver import (
    BoundaryConditions, ibvp_solver, eval_solver,
)
from lucifex.utils import SpatialPerturbation, cubic_noise
from lucifex.sim import configure_simulation

from .navier_stokes import (
    ipcs_solvers,
    advection_diffusion,
    newtonian_stress,
)


@configure_simulation(
    store_step=1,
    write_step=None,
)
def darcy_brinkman_rectangle(    
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str = 'quadrilateral',
    # physical
    Pr: float = 1.0,
    Ra: float = 1e7,
    Da: float = 1e-4,
    # initial perturbation
    noise_eps: float = 1e-6,
    noise_freq: tuple[int, int] = (8, 8),
    noise_seed: tuple[int, int] = (12, 34),
    # time step
    dt_max: float = 0.5,
    dt_min: float = 0.0,
    cfl_courant: float = 0.5,
    # time discretization
    D_adv_ns: FiniteDifference = FE,
    D_visc_ns: FiniteDifference = CN,
    D_buoy_ns: FiniteDifference = FE,
    D_darcy_ns: FiniteDifference = BE,
    D_adv_ad: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (BE, BE),
    D_diff_ad: FiniteDifference = CN,
):
    # time
    order = finite_difference_order(
        D_adv_ns, D_visc_ns, D_buoy_ns, D_adv_ad, D_diff_ad,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')

    # space
    Lx = 2.0
    Ly = 1.0
    Omega = rectangle_mesh(Lx, Ly, Nx, Ny, cell=cell)
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
    Da = Constant(Omega, Da, 'Da')

    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 1 - x[1],
        cubic_noise(['dirichlet', 'dirichlet'], [Lx, Ly], noise_freq, noise_seed),
        [Lx, Ly],
        noise_eps,
    )
    c_bcs = BoundaryConditions(
        ('dirichlet', dOmega['lower'], 1.0),
        ('dirichlet', dOmega['upper'], 0.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )  
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, u_zero),
    )  

    # flow
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    # transport
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    rho = ExprSeries(-c, 'rho')
    eg = as_vector([0, -1])
    chi = Function((Omega, 'DP', 0), name='chi')
    chi.interpolate(lambda x: 1.0 - 1.0 * (x[1] > Ly / 2))
    f = D_buoy_ns(rho * eg) - D_darcy_ns(u) * chi * sqrt(Pr / Ra) / Da

    # solvers
    dt_solver = eval_solver(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )
    ns_solvers = ipcs_solvers(
        u, p, dt[0], 1, sqrt(Pr/ Ra), newtonian_stress, D_adv_ns, D_visc_ns, f=f, u_bcs=u_bcs, 
    )
    c_solver = ibvp_solver(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, sqrt(Ra * Pr), D_adv_ad, D_diff_ad,
    )

    solvers = [dt_solver, *ns_solvers, c_solver]
    namespace = [Pr, Ra, Da, chi, rho]

    return solvers, t, dt, namespace