from ufl.core.expr import Expr
from ufl import dx, Form, grad, inner, TestFunction

from lucifex.fem import LUCiFExConstant as Constant, LUCiFExFunction as Function
from lucifex.fdm import DT, CN, FiniteDifference, ConstantSeries, FunctionSeries
from lucifex.mesh import interval_mesh
from lucifex.solver import BoundaryConditions, bvp_solver, dx_solver, eval_solver
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
    write_step=1, 
    write_file='series',
    dir_base='./data',
    dir_params=('Lx, Nx'),
)
def diffusion_simulation_1d(
    Lx: float, 
    Nx: int, 
    dt: float, 
    m_exponent: float,
    D_fdm: FiniteDifference,
):
    mesh = interval_mesh(Lx, Nx)
    u = FunctionSeries((mesh, 'P',  1), "u", D_fdm.order)
    t = ConstantSeries(mesh, 't', D_fdm.order)
    dt = Constant(mesh, dt, 'dt')
    bcs = BoundaryConditions(
        ('dirichlet', lambda x: x[0], 0.0)
        ('dirichlet', lambda x: x[0] - Lx, 0.0)
    )
    bvp = bvp_solver(diffusion, bcs)(u, dt, D_fdm)
    m = ConstantSeries(mesh, "m", D_fdm.order)
    m_integrand = lambda u, n: u ** n
    m_solver = dx_solver(m, m_integrand)(u[0], m_exponent)
    uMax = ConstantSeries(mesh, "uMinMax", D_fdm.order)
    uMax_solver = eval_solver(uMax, maximum(u[0]))
    solvers = [bvp, m_solver, uMax_solver]
    return solvers, t, dt