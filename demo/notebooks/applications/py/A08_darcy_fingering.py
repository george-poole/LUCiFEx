import numpy as np
import scipy.special as sp

from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import FiniteDifference, FiniteDifferenceArgwise, AB2, CN
from lucifex.fdm.ufl_operators import exp
from lucifex.fem import Constant, SpatialPerturbation, sinusoid_noise
from lucifex.solver import BoundaryConditions
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, BoundaryType
from lucifex.pde.scaling import ScalingOptions

from .A04_darcy_convection_generic import darcy_convection_generic


DARCY_FINGERING_SCALINGS = ScalingOptions(
    ('Ad', 'Di', 'In', 'Xl'),
    lambda Pe: (
        {'advective': (1, 1/Pe, 1, 1)},
        {'diffusive': (1, 1, Pe, 1)}
    )
)
"""
Choice of length scale `ℒ`, velocity scale `𝒰`
and time scale `𝒯` in the non-dimensionalization.

`'advective'` \\
`ℒ` = domain size \\
`𝒰` = injection speed

`'diffusive'` \\
`ℒ` = domain size \\
`𝒰` = diffusive speed
"""


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def darcy_fingering_rectangle(
    # domain
    aspect: float = 2.0,
    Ly: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    # physical
    scaling: str = 'advective',
    Pe: float = 5e2,
    Lmbda: float = 1e1,
    bc_type: BoundaryType = BoundaryType.DIRICHLET,
    erf_eps: float = 1e-2,
    c_eps: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    # time step
    dt_max: float = 0.25,
    cfl_h: str | float = "hmin",
    cfl_courant: float = 0.25,
   # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
):
    scaling_map = DARCY_FINGERING_SCALINGS[scaling](Pe)
    Xl = scaling_map['Xl']
    Lx = aspect * Xl
    Ly = 1.0 * Xl

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
    Di, In = scaling_map[Omega, 'Di', 'In']
    Di = Constant(Omega, Di, 'Di')
    In = Constant(Omega, In, 'In')
    Pe = Constant(Omega, Pe, 'Pe')
    Lmbda = Constant(Omega, Lmbda, 'Lmbda')

    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 0.5 * (1.0 + sp.erf(x[0] / (Lx * erf_eps))),
        sinusoid_noise(['neumann', 'neumann'], [Lx, Ly], c_freq),
        [Lx, Ly],
        c_eps,
    ) 
    if bc_type == BoundaryType.DIRICHLET:
        psi_bcs = BoundaryConditions(
            (BoundaryType.DIRICHLET, dOmega.union, 0.0),
        )
        c_bcs = BoundaryConditions(
            ("dirichlet", dOmega['left'], 0.0),
            ("dirichlet", dOmega['right'], 1.0),
            ("neumann", dOmega['upper', 'lower'], 0.0),
        )
    elif bc_type == BoundaryType.PERIODIC:
        right_to_left = lambda x: np.vstack((x[0] - Lx, x[1]))
        psi_bcs = BoundaryConditions(
            (BoundaryType.PERIODIC, dOmega['right'], right_to_left),
            ("dirichlet", dOmega['upper', 'lower'], 0.0),
        )
        c_bcs = BoundaryConditions(
            (BoundaryType.PERIODIC, dOmega['right'], right_to_left),
            ("neumann", dOmega['upper', 'lower'], 0.0),
        )
    else:
        raise ValueError
    
    # constitutive
    dispersion = lambda phi: Di * phi
    viscosity = lambda c: exp(-Lmbda * c)
    density = lambda c: In * viscosity(c)

    simulation = darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=1,
        egy=None,
        dispersion=dispersion,
        density=density,
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