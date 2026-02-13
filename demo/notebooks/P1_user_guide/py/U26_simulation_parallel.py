import time
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fem import Constant
from lucifex.fdm import CN, FunctionSeries, ConstantSeries
from lucifex.solver import ibvp, BoundaryConditions
from lucifex.sim import Simulation, configure_simulation, parallel_run
from lucifex.sim import create_and_run, parallel_run
from lucifex.pde.diffusion import diffusion


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def create_simulation(
    Nx: int,
    Ny: int,
    dt: float,
    d: float = 0.01,
) -> Simulation:
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
    t = ConstantSeries(mesh, name='t', ics=0.0)
    dt = Constant(mesh, dt, name='dt')
    d = Constant(mesh, d, name='d')
    ics = lambda x: np.exp(-((x[0] - Lx/2)**2 + (x[1] - Ly/2)**2)/ (0.01 * Lx))
    bcs = BoundaryConditions(
        ("dirichlet", boundary.union, 0.0),  
    )
    u = FunctionSeries((mesh, 'P', 1), name='u', store=1)
    u_solver = ibvp(diffusion, ics, bcs)(u, dt, d, CN)
    return u_solver, t, dt
    

def performance_report(
    t_exec: float,
    n_proc: int | None,
    n_stop: int,
    store: int,
    Nx: int,
    Ny: int,
) -> str:
    s = "\n".join(
        [
            f'Nx = {Nx}',
            f'Ny = {Ny}',
            f'store = {store}',
            f'n_stop = {n_stop}',
            f'n_proc = {n_proc}',
            f't_exec = {t_exec}',
        ]
    )
    return s


if __name__ == "__main__":
    STORE = 1
    N_PROC = 4
    N_STOP = 200
    STORE = 1
    NX = 200
    NY = 200
    DT = 0.01
    D_OPTS = (0.1, 1.0, 5.0, 10.0)
    create_sim = create_simulation(store_delta=STORE)
    parallel_run(
        create_sim, N_PROC, N_STOP, return_as='grid',
    )(NX, NY, DT)(d=D_OPTS)

