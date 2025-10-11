from lucifex.fdm import (
    FiniteDifference, cfl_timestep, 
    FunctionSeries, ConstantSeries, finite_difference_order,
)
from lucifex.fem import LUCiFExConstant as Constant
from lucifex.solver import (
    BoundaryConditions, eval_solver,
)
from lucifex.mesh import ellipse_obstacle_mesh, mesh_boundary
from lucifex.sim import configure_simulation

from.navier_stokes import ipcs_solvers, chorin_solvers, newtonian_stress


@configure_simulation(
    store_step=1,
    write_step=None,
)
def navier_stokes_circle_obstacle(
    # domain
    Lx: float,
    Ly: float,
    r: float,
    c: tuple[float, float],
    dx: float,
    # physical
    rho: float,
    mu: float,
    p_in: float,
    # time step
    dt_max: float,
    dt_min: float,
    cfl_courant: float,
    ns_scheme: str,
    # time discretization
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
):
    # time
    order = finite_difference_order(D_adv, D_visc)
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')

    # space
    Omega = ellipse_obstacle_mesh(dx, 'triangle')(Lx, Ly, r, c)
    dOmega = mesh_boundary(
        Omega,
        {
            'obstacle': lambda x: (x[0] - c[0]) **2 + (x[1] - c[1]) **2 - r**2,
            'upper': lambda x: x[1] - Ly,
            'lower': lambda x: x[1],
            'left': lambda x: x[0],
            'right': lambda x: x[1] - Lx,
        }
    )
    dim = Omega.geometry.dim
    u_zero = [0.0] * dim

    # constants
    rho = Constant(Omega, rho, 'rho')
    mu = Constant(Omega, mu, 'mu')

    # boundary conditions
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega['upper', 'lower', 'obstacle'], u_zero),
    )
    p_bcs = BoundaryConditions(
        ('dirichlet', dOmega['left'], p_in),
        ('dirichlet', dOmega['right'], 0.0),
    )

    # flow
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)

    # solvers
    dt_solver = eval_solver(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )
    if ns_scheme == 'ipcs':
        ns_solvers = ipcs_solvers(
            u, p, dt[0], rho, mu, newtonian_stress, D_adv, D_visc, u_bcs=u_bcs, p_bcs=p_bcs)
    elif ns_scheme == 'chorin':
        ns_solvers = chorin_solvers(u, p, dt[0], rho, mu, D_adv, D_visc, u_bcs=u_bcs, p_bcs=p_bcs)
    else:
        raise ValueError(f"Navier-Stokes scheme '{ns_scheme}' not implemented.")
    
    solvers = [dt_solver, *ns_solvers]
    namespace = [rho, mu]
    return solvers, t, dt, namespace
