from ufl import as_vector

from lucifex.fdm import FiniteDifference, FE, CN, BE
from lucifex.fem import Constant
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference,
        FiniteDifferenceArgwise, ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.solver import (
    BoundaryConditions, ibvp, evaluation,
)
from lucifex.utils import SpatialPerturbation, cubic_noise
from lucifex.sim import configure_simulation

from lucifex.pde.navier_stokes import ipcs_solvers
from lucifex.pde.constitutive import newtonian_stress
from lucifex.pde.advection_diffusion import advection_diffusion


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def navier_stokes_rayleigh_taylor_rectangle(
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str = 'quadrilateral',
    # physical
    Pr = 1.0,
    Ra = 1e3,
    # initial perturbation
    noise_eps: float = 1e-6,
    noise_freq: tuple[int, int] = (8, 8),
    noise_seed: tuple[int, int] = (12, 34),
    # time step
    dt_max: float = 0.5,
    dt_min: float = 0.0,
    cfl_courant: float = 0.75,
    # time discretization
    D_adv_ns: FiniteDifference = FE,
    D_visc_ns: FiniteDifference = CN,
    D_buoy_ns: FiniteDifference = FE,
    D_adv_ad: FiniteDifference | FiniteDifferenceArgwise = (BE @ BE),
    D_diff_ad: FiniteDifference = CN,
):
    """
    `∂c/∂t + 𝐮·∇c = ∇²c`

    `∇·𝐮 = 0`

    `∂𝐮/∂t + 𝐮·∇𝐮 = Pr(-∇p + ∇²𝐮) + PrRa ρ 𝐞₉`
    """
    # space
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
    u_zero = [0.0] * dim

    # time
    order = finite_difference_order(
        D_adv_ns, D_visc_ns, D_buoy_ns, D_adv_ad, D_diff_ad,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')


    # constants
    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')  

    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 1.0 * (x[1] > Ly / 2) + 0.0,
        cubic_noise(['neumann', 'neumann'], [Lx, Ly], noise_freq, noise_seed),
        [Lx, Ly],
        noise_eps,
    )   
    c_bcs = BoundaryConditions(
        ('neumann', dOmega.union, 0.0)
    )
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, u_zero),
    )

    # flow and transport
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    deviatoric_stress = lambda u: Pr * newtonian_stress(u, 1)
    rho = ExprSeries(c, 'rho')
    f = Pr * Ra * rho * as_vector([0, -1]) 

    # solvers
    dt_solver = evaluation(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )
    ns_solvers = ipcs_solvers(
        u, p, dt[0], deviatoric_stress, D_adv_ns, D_visc_ns, D_buoy_ns, f, u_bcs, p_coeff=Pr,
    )
    c_solver = ibvp(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, 1, D_adv_ad, D_diff_ad,
    )

    solvers = [dt_solver, *ns_solvers, c_solver]
    namespace = [Pr, Ra, rho]
    return solvers, t, dt, namespace

