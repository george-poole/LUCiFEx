import scipy.special as sp

from lucifex.fdm import (
    FunctionSeries, FiniteDifference, AB1, ConstantSeries, 
    finite_difference_order, ExprSeries, cfl_timestep,
)
from lucifex.fem import LUCiFExConstant as Constant
from lucifex.solver import(
    BoundaryConditions, OptionsPETSc, interpolation_solver,
    ibvp_solver, bvp_solver, eval_solver,
)
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, cubic_noise

from .porous import porous_advection_diffusion_reaction, darcy_streamfunction, streamfunction_velocity
from .utils import rectangle_domain


@configure_simulation(
    store_step=1,
    write_step=None,
)
def porous_abc_rectangle(
    # domain
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    # physical
    Lmbda: float = 100,
    delta_b: float = 1.0,
    delta_c: float = 1.0,
    beta: float = 1.0,
    gamma: float = 2.0,
    # initial conditions
    erf_eps: float = 1e-2,
    a_eps: float = 1e-6,
    a_freq: tuple[int, int] = (8, 8),
    a_seed: tuple[int, int] = (1234, 5678),
    b0: float = 1.0,
    c0: float = 0.0,
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
    psi_petsc: OptionsPETSc | None = None,
    abc_petsc: OptionsPETSc | None = None,   
):
    # time
    order = finite_difference_order(
        D_adv, D_diff, D_reac,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')

    # space
    Omega, dOmega = rectangle_domain(Lx, Ly, Nx, Ny, cell)

    # constants
    Lmbda = Constant(Omega, Lmbda, 'Lmbda')
    beta = Constant(Omega, beta, 'beta')
    gamma = Constant(Omega, gamma, 'gamma')
    delta_b = Constant(Omega, delta_b, 'delta_b')
    delta_c = Constant(Omega, delta_c, 'delta_c')

    # initial and boundary conditions
    a_ics = SpatialPerturbation(
        lambda x: 1 + sp.erf((x[1] - Ly) / (Ly * erf_eps)),
        cubic_noise(['neumann', ('neumann', 'dirichlet')], [Lx, Ly], a_freq, a_seed),
        [Lx, Ly],
        a_eps,
    ) 
    a_bcs = BoundaryConditions(
        ('dirichlet', dOmega['upper'], 1.0),
        ('neummann', dOmega['left', 'right', 'lower'], 0.0),
    )
    b_bcs = BoundaryConditions(
        ('neummann', dOmega.union, 0.0),
    )
    c_bcs = BoundaryConditions(
        ('neummann', dOmega.union, 0.0),
    )
    psi_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, 0.0),
    )

    # flow
    psi_deg = 2
    psi = FunctionSeries((Omega, 'P', psi_deg), 'psi')
    u = FunctionSeries((Omega, "P", psi_deg - 1, 2), "u", order)
    # transport
    a = FunctionSeries((Omega, 'P', 1), 'a', order, ics=a_ics)
    b = FunctionSeries((Omega, 'P', 1), 'b', order, ics=b0)
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c0)
    # constitutive
    rho = Lmbda * ExprSeries(a + beta * b + gamma * c, 'rho')
    r = ExprSeries(a * b, 'r')

    # solvers
    psi_solver = bvp_solver(darcy_streamfunction, psi_bcs, psi_petsc)(psi, 1, 1, rho[0], egy=-1)
    u_solver = interpolation_solver(u, streamfunction_velocity)(psi[0])
    dt_solver = eval_solver(dt, cfl_timestep)(
            u[0], cfl_h, cfl_courant, dt_max, dt_min,
        ) 

    a_solver = ibvp_solver(porous_advection_diffusion_reaction, bcs=a_bcs, petsc=abc_petsc)(
        a, dt, 1, u, 1, 1, 1, r, D_adv, D_diff, D_reac,
    )
    b_solver = ibvp_solver(porous_advection_diffusion_reaction, bcs=b_bcs, petsc=abc_petsc)(
        a, dt, 1, u, 1, delta_b, 1, r, D_adv, D_diff, D_reac,
    )
    c_solver = ibvp_solver(porous_advection_diffusion_reaction, bcs=c_bcs, petsc=abc_petsc)(
        a, dt, 1, u, 1, delta_c, 1, r, D_adv, D_diff, D_reac,
    )

    solvers = [psi_solver, u_solver, dt_solver, a_solver, b_solver, c_solver]
    namespace = [Lmbda, beta, gamma, delta_b, delta_c]
    return solvers, t, dt, namespace