import numpy as np
import scipy.special as sp
from typing import Callable
from ufl import cos, sin

from lucifex.mesh import mesh_boundary, rectangle_mesh
from lucifex.fdm import FiniteDifference, FiniteDifferenceArgwise, AB2, CN
from lucifex.fem import Constant, SpatialPerturbation, cubic_noise
from lucifex.solver import BoundaryConditions, OptionsPETSc
from lucifex.sim import Simulation, configure_simulation
from lucifex.utils.fenicsx_utils import CellType
from lucifex.pde.constitutive import permeability_cross_bedded

from .C10_darcy_convection_generic import darcy_convection_generic, DARCY_CONVECTION_SCALINGS


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
    # physical
    scaling: str = 'advective',
    Ra: float = 5e2,
    beta: float = 0.0,
    # constitutive relations
    porosity: Callable[[np.ndarray], np.ndarray] | float = 1,
    kappa: float = 1.0,
    vartheta: float = 0.0,
    # initial conditions
    c_ampl: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_seed: tuple[int, int] = (1234, 5678),
    c_eps: float = 1e-2,
    # time step
    dt_max: float = 0.5,
    dt_h: str | float = "hmin",
    dt_courant: float = 0.75,
    # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
    # linear algebra
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    # optional solvers
    diagnostic: bool = False,
) -> Simulation:
    # space
    scaling_map = DARCY_CONVECTION_SCALINGS[scaling](Ra)
    X = scaling_map['X']
    Lx = aspect * X
    Ly = 1.0 * X
    Omega = rectangle_mesh(Lx, Ly, Nx, Ny, cell)
    dOmega = mesh_boundary(
        Omega, 
        {
            "left": lambda x: x[0],
            "right": lambda x: x[0] - Lx,
            "lower": lambda x: x[1],
            "upper": lambda x: x[1] - Ly,
        },
    )
    # constants
    Di, Bu = scaling_map[Omega, 'Di', 'Bu']
    Ra = Constant(Omega, Ra, 'Ra')
    kappa = Constant(Omega, kappa, name='kappa')
    vartheta = Constant(Omega, np.radians(vartheta), name='vartheta')
    beta = Constant(Omega, np.radians(beta), name='beta')
    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 1 + sp.erf((x[1] - Ly) / (Ly * c_eps)),
        cubic_noise(['neumann', ('neumann', 'dirichlet')], [Lx, Ly], c_freq, c_seed),
        [Lx, Ly],
        c_ampl,
    ) 
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['upper'], 1.0),
        ('neumann', dOmega['lower', 'left', 'right'], 0.0)
    )
    # constitutive relations
    permeability = lambda phi: permeability_cross_bedded(phi**2, kappa, vartheta)
    dispersion = lambda phi: Di * phi
    density = lambda c: Bu * c
    egx = -sin(beta)
    egy = -cos(beta)
    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=egx,
        egy=egy, 
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        porosity=porosity,
        permeability=permeability,
        dispersion=dispersion,
        density=density, 
        dt_max=dt_max, 
        dt_h=dt_h, 
        dt_courant=dt_courant,
        D_adv=D_adv, 
        D_diff=D_diff, 
        psi_petsc=psi_petsc, 
        c_petsc=c_petsc, 
        diagnostic=diagnostic,
        auxiliary=(Ra, Di, Bu, beta, kappa, vartheta, ('Lx', Lx), ('Ly', Lx)),
    )