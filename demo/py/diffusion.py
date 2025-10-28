import numpy as np
from ufl import dx, Form, grad, inner, TestFunction

from lucifex.fem import Constant
from lucifex.fdm import DT, FiniteDifference, ConstantSeries, FunctionSeries
from lucifex.mesh import interval_mesh
from lucifex.solver import BoundaryConditions, ibvp, integration, evaluation
from lucifex.sim import configure_simulation
from lucifex.utils import maximum


def diffusion(
    u: FunctionSeries,
    dt: Constant,
    k: Constant,
    D_fdm: FiniteDifference,
) -> tuple[Form, Form]:
    v = TestFunction(u.function_space)
    Fdt = v * DT(u, dt) * dx
    Fdiff = inner(grad(v), k * grad(D_fdm(u))) * dx
    return Fdt, Fdiff


@configure_simulation(
    store_delta=1,
    write_delta=None, 
    write_file='FunctionSeries',
    dir_base='./data',
    dir_params=('Lx', 'Nx'),
)
def diffusion_simulation_1d(
    Lx: float, 
    Nx: int, 
    dt: float, 
    k: float,
    D_fdm: FiniteDifference,
    m_exponent: float,
):
    """
    Solves `∂u/∂t = ∇²c`. \\
    Evaluates `maxₓ(u)`, `∫ uᵐ dx`.
    """
    # time
    order = max(D_fdm.order, 2)
    t = ConstantSeries(mesh, 't', order, ics=0.0)
    dt = Constant(mesh, dt, 'dt')

    # space
    mesh = interval_mesh(Lx, Nx)
    
    # constants
    k = Constant(mesh, k, 'k')

    # PDE solver
    u = FunctionSeries((mesh, 'P',  1), "u", order)
    ics = lambda x: np.exp(-(x[0] - Lx/2)**2 / (0.01 * Lx))
    bcs = BoundaryConditions(
        ('dirichlet', lambda x: x[0], 0.0),
        ('dirichlet', lambda x: x[0] - Lx, 0.0),
    )
    u_solver = ibvp(diffusion, ics, bcs)(u, dt, k, D_fdm)

    # secondary solvers
    m = ConstantSeries(mesh, "m", order)
    m_integrand = lambda u, n: u ** n
    m_solver = integration(m, m_integrand)(u[0], m_exponent)
    uMax = ConstantSeries(mesh, "uMax", order)
    uMax_solver = evaluation(uMax, maximum)(u[0])

    solvers = [u_solver, m_solver, uMax_solver]
    return solvers, t, dt