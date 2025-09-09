from ufl.core.expr import Expr
from ufl import dx, Form, grad, inner

from lucifex.fem import LUCiFExConstant as Constant, LUCiFExFunction as Function
from lucifex.fdm import DT, CN, FiniteDifference, ConstantSeries, FunctionSeries
from lucifex.mesh import interval_mesh
from lucifex.solver import BoundaryConditions, bvp_solver, dx_solver, eval_solver
from lucifex.sim import create_simulation
from lucifex.utils import maximum


def diffusion_forms(
    u: FunctionSeries,
    dt: Constant,
    Dfdm: FiniteDifference,
) -> list[Form]:
    v = u.testfunction
    Fdt = v * DT(u, dt) * dx
    Fdiff += inner(grad(v), grad(Dfdm(u))) * dx
    return [Fdt, Fdiff]


def integrand(
    u: Function,
    n: float,
) -> Expr:
    return u ** n


@create_simulation(
    write_step=1, 
    write_file='series',
    dir_base='./data',
    dir_labels=('Lx, Nx'),
)
def diffusion_1d(
    Lx: float, 
    Nx: int, 
    dt: float, 
    Dfdm: FiniteDifference,
):
    mesh = interval_mesh(Lx, Nx)
    u = FunctionSeries((mesh, 'P',  1), "u", Dfdm.order)
    t = ConstantSeries(mesh, 't', Dfdm.order)
    dt = Constant(mesh, dt, 'dt')
    bcs = BoundaryConditions(
        ('dirichlet', lambda x: x[0], 0.0)
        ('dirichlet', lambda x: x[0] - Lx, 0.0)
    )
    bvp = bvp_solver(diffusion_forms, bcs)(u, dt, Dfdm)
    m = ConstantSeries(mesh, "m", Dfdm.order)
    m_solver = dx_solver(m, integrand)(u[0], 2)
    uMax = ConstantSeries(mesh, "uMinMax", Dfdm.order)
    uMax_solver = eval_solver(uMax, maximum(u[0]))
    solvers = [bvp, m_solver, uMax_solver]
    return solvers, t, dt