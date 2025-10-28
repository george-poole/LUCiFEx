import numpy as np
import scipy.special as sp
from typing import Callable
from ufl import cos, sin

from lucifex.mesh import mesh_boundary, rectangle_mesh
from lucifex.fdm import FiniteDifference, AB2, CN
from lucifex.fem import Constant
from lucifex.solver import BoundaryConditions, OptionsPETSc
from lucifex.sim import Simulation, configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, cubic_noise

from lucifex.pde.constitutive import permeability_cross_bedded
from .darcy_convection_generic import darcy_convection_generic



@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def darcy_convection_evolving_rectangle(
    # domain
    aspect: float = 2.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    # gravity
    beta: float = 0.0,
    # physical
    Ra: float = 5e2,
    # constitutive relations
    porosity: Callable[[np.ndarray], np.ndarray] | float = 1,
    kappa: float = 1.0,
    vartheta: float = 0.0,
    # initial conditions
    erf_eps: float = 1e-2,
    c_eps: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_seed: tuple[int, int] = (1234, 5678),
    # time step
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    cfl_courant: float = 0.75,
    # time discretization
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (AB2, CN),
    D_diff: FiniteDifference = CN,
    # linear algebra
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    # optional solvers
    secondary: bool = False,
) -> Simulation:
    Ly = Ra
    Lx = aspect * Ly
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
        ("dirichlet", dOmega['upper'], 1.0),
        ('neumann', dOmega['lower', 'left', 'right'], 0.0)
    )
    c_ics = SpatialPerturbation(
        lambda x: 1 + sp.erf((x[1] - Ly) / (Ly * erf_eps)),
        cubic_noise(['neumann', ('neumann', 'dirichlet')], [Lx, Ly], c_freq, c_seed),
        [Lx, Ly],
        c_eps,
    ) 

    kappa = Constant(Omega, kappa, name='kappa')
    vartheta_rad = vartheta * np.pi / 180
    vartheta = Constant(Omega, vartheta_rad, name='vartheta')
    permeability = lambda phi: permeability_cross_bedded(phi**2, kappa, vartheta)
    density = lambda c: c

    beta_rad = beta * np.pi / 180
    beta = Constant(Omega, beta_rad, name='beta')
    egx = -sin(beta)
    egy = -cos(beta)

    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=egx,
        egy=egy,
        Ad=1,
        Pe=1, 
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        porosity=porosity,
        permeability=permeability,
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