import scipy.special as sp

from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import FiniteDifference, AB2, CN
from lucifex.fdm.ufl_operators import exp
from lucifex.fem import LUCiFExConstant as Constant
from lucifex.solver import BoundaryConditions
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, sinusoid_noise
from .porous import porous_convection_simulation


@configure_simulation(
    store_step=1,
    write_step=None,
)
def saffman_taylor_rectangle(
    Lx: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    Pe: float = 5e2,
    Lmbda: float = 1e1,
    erf_eps: float = 1e-2,
    c_eps: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    dt_max: float = 0.25,
    cfl_h: str | float = "hmin",
    cfl_courant: float = 0.25,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (AB2, CN),
    D_diff: FiniteDifference = CN,
):
    # time

    
    # space
    Omega = rectangle_mesh(
        (-0.5 * Lx, 0.5 * Lx), 
        (0, Ly), 
        Nx, 
        Ny, 
        cell=cell,
    )
    dOmega = mesh_boundary(
        Omega,
        {
            "left": lambda x: x[0] + 0.5 * Lx,
            "right": lambda x: x[0] - 0.5 * Lx,
            "lower": lambda x: x[1],
            "upper": lambda x: x[1] - Ly,
        },
    )

    # constants
    Lmbda = Constant(Omega, Lmbda, 'Lmbda')

    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 0.5 * (1.0 + sp.erf(x[0] / (Lx * erf_eps))),
        sinusoid_noise(['neumann', 'neumann'], [Lx, Ly], c_freq),
        [Lx, Ly],
        c_eps,
    ) 
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['left'], 0.0),
        ("dirichlet", dOmega['right'], 1.0),
        ("neumann", dOmega['upper', 'lower'], 0.0),
    )
    psi_bcs = BoundaryConditions(
        ("dirichlet", dOmega.union, 0.0),
    )
    
    # constitutive
    viscosity = lambda c: exp(-Lmbda * c)

    simulation = porous_convection_simulation(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=1,
        egy=None,
        Ra=Pe,
        density=viscosity,
        viscosity=viscosity,
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        psi_bcs=psi_bcs,
        dt_max=dt_max, 
        cfl_h=cfl_h, 
        cfl_courant=cfl_courant,
        D_adv=D_adv,
        D_diff=D_diff,
    )
    return simulation