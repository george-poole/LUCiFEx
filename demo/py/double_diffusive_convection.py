from ufl import dx, Form, as_vector, TestFunction

from lucifex.fdm import DT, FiniteDifference, FE, CN, BE
from lucifex.fem import LUCiFExConstant as Constant
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference,
    ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import (
    BoundaryConditions, bvp_solver, ibvp_solver, eval_solver,
)
from lucifex.utils import SpatialPerturbation, cubic_noise
from lucifex.sim import configure_simulation

from .navier_stokes import (
    ipcs_1, ipcs_2, ipcs_3,
    newtonian_stress,
)


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
        F_neumann = sum([-(1 / Le) * v * cN * ds(i) for i, cN in c_neumann])
        forms.append(F_neumann)

    return forms


@configure_simulation(
    store_step=1,
    write_step=None,
)
def navier_stokes_double_diffusion(
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str = 'quadrialteral',
    # physical
    Pr = 1.0,
    Ra = 1e3,
    Rb = 1e3,
    Le = 1.0,
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
    D_force_ns: FiniteDifference = FE,
    D_adv_dd: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (BE, BE),
    D_diff_dd: FiniteDifference = CN,
):
    order = finite_difference_order(
        D_adv_ns, D_visc_ns, D_force_ns, D_adv_dd, D_diff_dd,
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
        ("dirichlet", dOmega['lower'], 0.0),
        ("dirichlet", dOmega['upper'], 1.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )
    c_ics = SpatialPerturbation(
        lambda x: x[1] / Ly,
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], noise_freq, noise_seed[:2]),
        [Lx, Ly],
        noise_eps,
        )   
    
    theta_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], 1.0),
        ("dirichlet", dOmega['upper'], 0.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )
    theta_ics = SpatialPerturbation(
        lambda x: 1 - x[1] / Ly,
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], noise_freq, noise_seed[2:]),
        [Lx, Ly],
        noise_eps,
        ) 

    t = ConstantSeries(Omega, 't', ics=0.0)
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    theta = FunctionSeries((Omega, 'P', 1), 'theta', order, ics=theta_ics)
    dt = ConstantSeries(Omega, 'dt')

    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')
    Rb = Constant(Omega, Rb, 'Rb')
    Le = Constant(Omega, Le, 'Le')
    rho = ExprSeries(Ra * c - Rb * theta, 'rho')
    f = rho * as_vector([0, -1]) 

    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, zero),
    )  

    dt_solver = eval_solver(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )

    ipcs1_solver = ibvp_solver(ipcs_1, bcs=u_bcs)(
        u, p, dt[0], 1/Pr, 1, newtonian_stress, D_adv_ns, D_visc_ns, D_force_ns, f,
    )
    ipcs2_solver = bvp_solver(ipcs_2, future=True)(
        p, u, dt[0], 1/Pr,
    )
    ipcs3_solver = bvp_solver(ipcs_3, future=True, overwrite=True)(
        u, p, dt[0], 1/Pr,
    )

    c_solver = ibvp_solver(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, Le, D_adv_dd, D_diff_dd,
    )
    theta_solver = ibvp_solver(advection_diffusion, bcs=theta_bcs)(
        theta, dt[0], u, 1, D_adv_dd, D_diff_dd,
    )

    solvers = [dt_solver, ipcs1_solver, ipcs2_solver, ipcs3_solver, c_solver, theta_solver]
    namespace = [Pr, Ra, Rb, Le, rho]

    return solvers, t, dt, namespace