from typing import Callable

from ufl import (
    dx, ds, Form, FacetNormal,
    TrialFunction, TestFunction, Identity, sym,
)
from ufl.core.expr import Expr

from lucifex.fdm import (
    DT, FE, CN, FiniteDifference, cfl_timestep, 
    FunctionSeries, ConstantSeries, Series, finite_difference_order,
)
from lucifex.fdm.ufl_operators import inner, div, nabla_grad, dot, grad
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.solver import (
    BoundaryConditions, bvp_solver, ibvp_solver, eval_solver,
)
from lucifex.mesh import ellipse_obstacle_mesh
from lucifex.sim import configure_simulation


def ipcs_1(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: ConstantSeries | Constant,
    rho: Constant,
    strain: Callable[[Function], Expr],
    stress: Callable[[Function, Function], Expr],
    D_adv: FiniteDifference,
    D_visc: FiniteDifference,
    D_force: FiniteDifference = FE,
    f: FunctionSeries | Function | Constant| None = None,
) -> list[Form]:
    v = TestFunction(u.function_space)
    n = FacetNormal(u.function_space.mesh)
    epsilon = strain(v)
    sigma = stress(D_visc(u), p[0])

    F_dudt = rho * inner(v, DT(u, dt)) * dx
    F_adv = rho * inner(v, D_adv(dot(u, nabla_grad(u)))) * dx
    F_visc = inner(epsilon, sigma) * dx
    F_ds = -inner(v, dot(n, sigma)) * ds

    forms = [F_dudt, F_adv, F_visc, F_ds]

    if f is not None:
        if isinstance(f, Series):
            f = D_force(f)
        F_f = -inner(v, f) * dx
        forms.append(F_f)

    return forms


def ipcs_2(
    p: FunctionSeries,
    u: FunctionSeries,
    dt: Constant,
    rho: Constant,
) -> tuple[Form, Form]:
    p_trial = TrialFunction(p.function_space)
    q = TestFunction(p.function_space)
    F_trial = inner(grad(q), grad(p_trial)) * dx
    F_grad = -inner(grad(q), grad(p[0])) * dx
    F_div = q * rho * (1 / dt) * div(u[1])  * dx
    return F_trial, F_grad, F_div


def ipcs_3(
    u: FunctionSeries,
    p: FunctionSeries,
    dt: Constant,
    rho: Constant,
) -> tuple[Form, Form]:
    u_trial = TrialFunction(u.function_space)
    v = TestFunction(u.function_space)
    F_dudt = rho * (1 / dt) * inner(v, (u_trial - u[1])) * dx
    F_grad = inner(v, grad(p[1]) - grad(p[0])) * dx
    return F_dudt, F_grad


@configure_simulation(
    store_step=1,
    write_step=None,
)
def navier_stokes_obstacle(
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
    scheme: str,
    # time discretization
    Dadv: FiniteDifference,
    Dvisc: FiniteDifference,
):
    order = finite_difference_order(Dadv, Dvisc)
    mesh = ellipse_obstacle_mesh(dx, 'triangle')(Lx, Ly, r, c)
    dim = mesh.geometry.dim
    zero = [0.0] * dim

    t = ConstantSeries(mesh, 't', ics=0.0)
    u = FunctionSeries((mesh, 'P', 2, dim), 'u', order, ics=zero)
    p = FunctionSeries((mesh, 'P', 1), 'p', order, ics=0.0)
    dt = ConstantSeries(mesh, 'dt')
    rho = Constant(mesh, rho, 'rho')
    mu = Constant(mesh, mu, 'mu')

    strain = lambda u: sym(nabla_grad(u))
    stress = lambda u, p: -p * Identity(dim) + 2 * mu * strain(u)

    obstacle = lambda x: (x[0] - c[0]) **2 + (x[1] - c[1]) **2 - r**2
    u_bcs = BoundaryConditions(
        ('dirichlet', lambda x: x[1], zero),
        ('dirichlet', lambda x: x[1] - Ly, zero),
        ('dirichlet', obstacle, zero),
    )
    p_bcs = BoundaryConditions(
        ('dirichlet', lambda x: x[0], p_in),
        ('dirichlet', lambda x: x[0] - Lx, 0.0),
    )

    dt_solver = eval_solver(dt, cfl_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )

    if scheme == 'ipcs':
        ipcs1_solver = ibvp_solver(ipcs_1, bcs=u_bcs)(
            u, p, dt[0], rho, strain, stress, FE, CN,
        )
        ipcs2_solver = bvp_solver(ipcs_2, bcs=p_bcs, future=True)(
            p, u, dt[0], rho,
        )
        ipcs3_solver = bvp_solver(ipcs_3, future=True, overwrite=True)(
            u, p, dt[0], rho,
        )
        ns_solvers = [ipcs1_solver, ipcs2_solver, ipcs3_solver]
    elif scheme == 'chorin':
        chorin1_solver = ...
        chorin2_solver = ...
        chorin3_solver = ...
        ns_solvers = [chorin1_solver, chorin2_solver, chorin3_solver]
        raise NotImplementedError
    else:
        raise ValueError(f'{scheme} not recognised')
    
    solvers = [dt_solver, *ns_solvers]
    namespace = [rho, mu]

    return solvers, t, dt, namespace
