from ufl import as_vector

from lucifex.fdm import FiniteDifference, FE, CN, BE
from lucifex.fem import LUCiFExConstant as Constant
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
    newtonian_stress,
    advection_diffusion,
)


@configure_simulation(
    store_step=1,
    write_step=None,
)
def navier_stokes_double_diffusive_rectangle(
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str = 'quadrilateral',
    # physical
    Le = 1.0,
    Pr = 1.0,
    Ra = 1e3,
    beta = 1.0,
    # initial perturbation
    noise_eps: float = 1e-6,
    noise_freq: tuple[int, int] = (8, 8),
    noise_seed: tuple[int, int, int, int] = (12, 34, 43, 21),
    # time step
    dt_max: float = 0.5,
    dt_min: float = 0.0,
    cfl_courant: float = 0.75,
    # time discretization
    D_adv_ns: FiniteDifference = FE,
    D_visc_ns: FiniteDifference = CN,
    D_buoy_ns: FiniteDifference = FE,
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
    Le = Constant(Omega, Le, 'Le')  
    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')
    beta = Constant(Omega, beta, 'beta')

    # initial conditions
    c_ics = SpatialPerturbation(
        lambda x: x[1] / Ly,
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], noise_freq, noise_seed[:2]),
        [Lx, Ly],
        noise_eps,
    )
    theta_ics = SpatialPerturbation(
        lambda x: 1 - x[1] / Ly,
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], noise_freq, noise_seed[2:]),
        [Lx, Ly],
        noise_eps,
    )  

    # boundary conditions
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], 0.0),
        ("dirichlet", dOmega['upper'], 1.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )
    theta_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], 1.0),
        ("dirichlet", dOmega['upper'], 0.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, u_zero),
    )  

    # flow
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    # transport
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    theta = FunctionSeries((Omega, 'P', 1), 'theta', order, ics=theta_ics)
    # constitutive
    rho = ExprSeries(c - beta * theta, 'rho')
    eg = as_vector([0, -1])
    f = Ra * rho * eg

    # solvers
    dt_solver = eval_solver(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )
    ns_solvers = ipcs_solvers(
        u, p, dt[0], 1/Pr, 1, newtonian_stress, D_adv_ns, D_visc_ns,  D_buoy_ns, f, u_bcs,
    )
    c_solver = ibvp_solver(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, 1, D_adv_ad, D_diff_ad,
    )
    theta_solver = ibvp_solver(advection_diffusion, bcs=theta_bcs)(
        theta, dt[0], u, 1/Le, D_adv_ad, D_diff_ad,
    )

    solvers = [dt_solver, *ns_solvers, c_solver, theta_solver]
    namespace = [Le, Pr, Ra, beta, rho]
    return solvers, t, dt, namespace