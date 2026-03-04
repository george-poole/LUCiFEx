import numpy as np
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fem import Constant
from lucifex.fdm import CN, FunctionSeries, ConstantSeries
from lucifex.solver import ibvp, BoundaryConditions
from lucifex.sim import Simulation, configure_simulation, parallel_run
from lucifex.pde.diffusion import diffusion


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def create_simulation(
    Nx: int = 10,
    Ny: int = 10,
    dt: float = 0.01,
    d: float = 1.0,
) -> Simulation:
    # space
    Lx, Ly = 1.0, 1.0
    mesh = rectangle_mesh(Lx, Ly, Nx, Ny)
    boundary = mesh_boundary(
        mesh, 
        {
            "left": lambda x: x[0],
            "right": lambda x: x[0] - Lx,
            "lower": lambda x: x[1],
            "upper": lambda x: x[1] - Ly,
        },
    )
    # time
    t = ConstantSeries(mesh, name='t', ics=0.0)
    dt = Constant(mesh, dt, name='dt')
    # constant
    d = Constant(mesh, d, name='d')
    # initial and boundary conditions
    ics = lambda x: np.exp(-((x[0] - Lx/2)**2 + (x[1] - Ly/2)**2)/ (0.01 * Lx))
    bcs = BoundaryConditions(
        ("dirichlet", boundary.union, 0.0),  
    )
    # solver
    u = FunctionSeries((mesh, 'P', 1), name='u', store=1)
    u_solver = ibvp(diffusion, ics, bcs)(u, dt, d, CN)
    return u_solver, t, dt