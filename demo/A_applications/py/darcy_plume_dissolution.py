import numpy as np
from ufl.core.expr import Expr

from lucifex.fdm import FiniteDifference
from lucifex.fem import Function as Function, SpatialConstant as Constant
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, AB1,
    ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, bvp, ibvp, 
    evaluation, interpolation
)
from lucifex.utils import CellType
from lucifex.sim import configure_simulation

from lucifex.pde.transport import advection_diffusion, advection_diffusion_reaction
from lucifex.pde.darcy import darcy_incompressible
from lucifex.pde.evolution import evolution_expression


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def darcy_plume_dissolution_rectangle(
    # domain
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str = CellType.TRIANGLE,
    # physical
    Bu: float = 1000.0,
    Ki: float = 1e3,
    Le: float = 1.0,
    epsilon: float = 1e-2,
    beta: float = 1.0,
    # inflow
    delta: float = 0.1,
    b_in: float = 0.0,
    # initial conditions
    c0: float = 1.0,
    b0: float = 0.0,
    s0: float = 0.1,
    # time step
    dt_min: float = 0.0,
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    cfl_courant: float | None = 0.75,
    # time discretization
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = AB1,
    D_diff: FiniteDifference = AB1,
    D_reac: FiniteDifference = AB1,
    D_evol: FiniteDifference = AB1,
    # linear algebra
    up_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    b_petsc: OptionsPETSc | None = None,
):
    # space
    Omega = rectangle_mesh((-Lx/2, Lx/2), Ly, Nx, Ny, cell=cell)
    dOmega = mesh_boundary(
        Omega, 
        {
            "left": lambda x: x[0] + 0.5 * Lx,
            "right": lambda x: x[0] - 0.5 * Lx,
            "inflow": lambda x: np.isclose(x[1], 0.0) & ((x[0] <= 0.5 * delta * Lx) & (x[0] >= -0.5 * delta * Lx)),
            "lower": lambda x: np.isclose(x[1], 0.0) & ((x[0] > 0.5 * delta * Lx) | (x[0] < -0.5 * delta * Lx)),
            "upper": lambda x: x[1] - Ly,
        },
    )

    # time
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(Omega, "t", order, ics=0.0)  
    dt = ConstantSeries(Omega, 'dt')

    # boundary conditions
    c_bcs = BoundaryConditions(
        ('neumann', dOmega['left', 'right', 'upper', 'lower'], 0.0),
        ('dirichlet', dOmega['inflow'], 0.0),
    )
    b_bcs = BoundaryConditions(
        ('neumann', dOmega['left', 'right', 'upper', 'lower'], 0.0),
        ('dirichlet', dOmega['inflow'], b_in),

    )
    u_bcs = BoundaryConditions(
        ('essential', dOmega['left', 'right', 'lower'], (0.0, 0.0), 0),
        ('essential', dOmega['upper'], (0.0, delta), 0),
        ('essential', dOmega['inflow'], (0.0, 1.0), 0),
    )

    # constants 
    Bu = Constant(Omega, Bu, 'Bu')
    Ki = Constant(Omega, Ki, 'Ki')
    Le = Constant(Omega, Le, 'Le')
    epsilon = Constant(Omega, epsilon, 'epsilon')
    beta = Constant(Omega, beta, 'beta')
    egx = 0
    egy = -1

    # flow
    u_fam = 'BDMCF' if Omega.topology.cell_name() == CellType.QUADRILATERAL else 'BDM'
    u_deg = 1
    up = FunctionSeries((Omega, [(u_fam, u_deg), ('DP', u_deg - 1)]), "up", order)
    u = up.sub(0, 'u')
    # transport
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c0)
    b = FunctionSeries((Omega, 'P', 1), 'b', order, ics=b0)
    # evolution
    s = FunctionSeries((Omega, 'P', 1), 's', order, ics=s0)
    # constitutive
    varphi = 1
    phi = ExprSeries(varphi * (1 - c), 'phi')
    k = ExprSeries(phi**2, 'k')
    rho = ExprSeries(c + beta * b, 'rho')
    r = Expr(s * (1 - c), 'r')

    # solvers
    up_petsc = OptionsPETSc("gmres", "lu") if up_petsc is None else up_petsc
    up_petsc['pc_factor_mat_solver_type'] = 'mumps'
    up_solver = bvp(darcy_incompressible, u_bcs, petsc=up_petsc)(
        up, Bu * rho[0], k[0], 1, egx, egy,
    )
    dt_solver = evaluation(dt, cfl_timestep)(
            u[0], cfl_h, cfl_courant, dt_max, dt_min,
        ) 

    c_solver = ibvp(advection_diffusion_reaction, bcs=c_bcs, petsc=c_petsc)(
        c, dt[0], u, 1, Ki * r, D_adv, D_diff, D_reac, phi=phi,
    )
    b_solver = ibvp(advection_diffusion, bcs=b_bcs, petsc=b_petsc)(
        c, dt[0], phi, u, Bu, 1, D_adv, D_diff, phi=phi,
    )
    s_solver = interpolation(s, evolution_expression)(
        s, dt[0], -epsilon * Ki * (1 / varphi) * r , D_evol,
    )

    solvers = [up_solver, dt_solver, c_solver, b_solver, s_solver]
    namespace = [Bu, Ki, Le, epsilon, beta, phi, k, rho, r]
    return solvers, t, dt, namespace