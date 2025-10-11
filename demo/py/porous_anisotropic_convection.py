import numpy as np
import scipy.special as sp
from typing import Callable
from ufl import cos, sin, as_tensor

from lucifex.fdm import FiniteDifference, AB2, CN
from lucifex.fem import LUCiFExConstant as Constant
from lucifex.solver import BoundaryConditions, OptionsPETSc
from lucifex.sim import Simulation, configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, cubic_noise

from .porous import porous_convection_simulation
from .utils import rectangle_domain


def permeability_cross_bedded(
    Kphi,
    kappa,
    vartheta,
):
    """
    `𝖪(ϕ) = K(ϕ) (
        (cos²ϑ + κsin²ϑ , (1 - κ)cosϑsinϑ), 
        ((1 - κ)cosϑsinϑ , κcos²ϑ + sin²ϑ), 
    )`
    """
    cs = cos(vartheta)
    sn = sin(vartheta)  
    tensor = as_tensor(
        (
            (cs**2 + kappa*sn**2, (1 - kappa)*cs*sn),
            ((1 - kappa)*cs*sn, kappa*cs**2 + sn**2), 
        ),
    )
    return Kphi * tensor


@configure_simulation(
    store_step=1,
    write_step=None,
)
def porous_anisotropic_convection_rectangle(
    # domain
    Lx: float = 2.0,
    Ly: float = 1.0,
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
    Omega, dOmega = rectangle_domain(Lx, Ly, Nx, Ny, cell)
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

    return porous_convection_simulation(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=egx,
        egy=egy,
        Ra=Ra, 
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