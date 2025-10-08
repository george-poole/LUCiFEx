import numpy as np
from ufl import dx, Form, grad, inner, TestFunction

from lucifex.fem import LUCiFExConstant as Constant
from lucifex.fdm import DT, FiniteDifference, ConstantSeries, FunctionSeries
from lucifex.mesh import interval_mesh
from lucifex.solver import BoundaryConditions, ibvp_solver, dx_solver, eval_solver
from lucifex.sim import configure_simulation
from lucifex.utils import maximum


def diffusion(
    u: FunctionSeries,
    dt: Constant,
    D_fdm: FiniteDifference,
) -> list[Form]:
    v = TestFunction(u.function_space)
    Fdt = v * DT(u, dt) * dx
    Fdiff = inner(grad(v), grad(D_fdm(u))) * dx
    return [Fdt, Fdiff]


@configure_simulation(
    store_step=None,
    write_step=None, 
    write_file='series',
    dir_base='./data',
    dir_params=('Lx', 'Nx'),
)
def diffusion_simulation_interval(
    Lx: float, 
    Nx: int, 
    dt: float, 
    m_exponent: float,
    D_fdm: FiniteDifference,
):
    order = max(D_fdm.order, 2)
    mesh = interval_mesh(Lx, Nx)
    u = FunctionSeries((mesh, 'P',  1), "u", order)
    t = ConstantSeries(mesh, 't', order, ics=0.0)
    dt = Constant(mesh, dt, 'dt')
    bcs = BoundaryConditions(
        ('dirichlet', lambda x: x[0], 0.0),
        ('dirichlet', lambda x: x[0] - Lx, 0.0),
    )
    ics = lambda x: np.exp(-(x[0] - Lx/2)**2 / (0.01 * Lx))
    u_solver = ibvp_solver(diffusion, ics, bcs)(u, dt, D_fdm)
    m = ConstantSeries(mesh, "m", order)
    m_integrand = lambda u, n: u ** n
    m_solver = dx_solver(m, m_integrand)(u[0], m_exponent)
    uMax = ConstantSeries(mesh, "uMax", order)
    uMax_solver = eval_solver(uMax, maximum)(u[0])
    solvers = [u_solver, m_solver, uMax_solver]
    return solvers, t, dt