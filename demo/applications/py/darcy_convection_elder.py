from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import FiniteDifference, FiniteDifferenceArgwise, AB2, CN
from lucifex.solver import BoundaryConditions, OptionsPETSc
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, cubic_noise

from .darcy_convection_generic import darcy_convection_generic


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def darcy_convection_elder_rectangle(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    Ra: float = 5e2,
    c_eps: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_seed: tuple[int, int] = (1234, 5678),
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    cfl_courant: float = 0.75,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    secondary: bool = False,
):
    Omega = rectangle_mesh(Lx, Ly, Nx, Ny, cell=cell)
    dOmega = mesh_boundary(
        Omega, 
        {
            "left": lambda x: x[0],
            "right": lambda x: x[0] - Lx,
            "lower": lambda x: x[1],
            "upper": lambda x: x[1] - Ly,
        },
    )
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], 0.0),
        ("dirichlet", dOmega['upper'], lambda x: 0.0 + 1.0 * (x[0] < Lx / 2)),
        ('neumann', dOmega['left', 'right'], 0.0)
    )
    c_ics = SpatialPerturbation(
        0.0,
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], c_freq, c_seed),
        [Lx, Ly],
        c_eps,
        ) 
    density = lambda c: c
    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        Pe=Ra, 
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        density=density, 
        dt_max=dt_max, 
        cfl_h=cfl_h, 
        cfl_courant=cfl_courant,
        D_adv=D_adv, 
        D_diff=D_diff, 
        psi_petsc=psi_petsc, 
        c_petsc=c_petsc, 
        secondary=secondary,
    )