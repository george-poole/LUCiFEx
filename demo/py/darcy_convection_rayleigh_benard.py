import numpy as np
from ufl import SpatialCoordinate, sqrt

from lucifex.mesh import rectangle_mesh, annulus_mesh, circle_sector_mesh, mesh_boundary
from lucifex.fem import Constant
from lucifex.fdm import FiniteDifference, AB2, CN
from lucifex.solver import BoundaryConditions, OptionsPETSc
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, cubic_noise

from .darcy_convection_generic import darcy_convection_generic


@configure_simulation(
    store_step=1,
    write_step=None,
)
def darcy_convection_rayleigh_benard_rectangle(
    # domain
    Lx: float = 2.0,
    Nx: int = 64,
    Ny: int = 64,
    cell: str = CellType.QUADRILATERAL,
    # physical
    Ra: float = 1e2,
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
    secondary: bool = False,
):
    """
    Non-dimensionalization choosing `ℒ` as rectangle depth, 
    `𝒰` as convective speed and `𝒯` constructed from `ℒ` and `𝒰`.

    `∂c/∂t + 𝐮·∇c = 1/Ra ∇²c`

    `∇⋅𝐮 = 0` \\
    `𝐮 = -(∇p + c𝐞ʸ)
    """
    Ly = 1.0
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
    Ra = Constant(Omega, Ra, 'Ra')
    c_ics = SpatialPerturbation(
        lambda x: 1 - x[1] / Ly,
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], c_freq, c_seed),
        [Lx, Ly],
        c_eps,
    )   
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], 1.0),
        ("dirichlet", dOmega['upper'], 0.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )
    dispersion = lambda phi: (1/Ra) * phi
    density = lambda c: -c
    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        dispersion=dispersion,
        density=density, 
        dt_max=dt_max, 
        cfl_h=cfl_h, 
        cfl_courant=cfl_courant,
        D_adv=D_adv, 
        D_diff=D_diff, 
        psi_petsc=psi_petsc, 
        c_petsc=c_petsc, 
        secondary=secondary,
        namespace_extras=[Ra],
    )


@configure_simulation(
    store_step=1,
    write_step=None,
)
def darcy_convection_rayleigh_benard_annulus(
    Rinner: float = 1.0,
    Router: float = 2.0,
    Nradial: int = 100,
    cell: str = CellType.TRIANGLE,
    Ra: float = 5e2,
    c_eps: float = 1e-6,
    c_freq: int = 8,
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    cfl_courant: float = 0.75,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (AB2, CN),
    D_diff: FiniteDifference = CN,
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    secondary: bool = False,
):
    """
    Non-dimensionalization choosing `ℒ` as annulus length scale, 
    `𝒰` as convective speed and `𝒯` constructed from `ℒ` and `𝒰`.
    """
    r2 = lambda x: x[0]**2 + x[1]**2
    r = lambda x, sqrt: sqrt(r2(x))
    dr = (Router - Rinner) / Nradial
    Omega = annulus_mesh(dr, cell)(Rinner, Router)
    dOmega = mesh_boundary(
        Omega, 
        {
            "inner": lambda x: r2(x) - Rinner**2,
            "outer": lambda x: r2(x) - Router**2,
        },
    )
    Ra = Constant(Omega, Ra, 'Ra')
    radial_noise = lambda x: c_eps * np.sin(c_freq * np.pi * (r(x, np.sqrt) - Rinner) / (Router - Rinner))
    c_ics = SpatialPerturbation(
        lambda x: np.log(Router / r(x, np.sqrt)) / np.log(Router / Rinner),
        radial_noise,
        Omega.geometry.x,
        c_eps,
        )  
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['inner'], 1.0),
        ("dirichlet", dOmega['outer'], 0.0),  
    ) 
    dispersion = lambda phi: (1/Ra) * phi
    density = lambda c: -c
    x = SpatialCoordinate(Omega)
    egx = -x[0] / r(x, sqrt)
    egy = -x[1] / r(x, sqrt)
    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=egx,
        egy=egy,
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        dispersion=dispersion,
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


@configure_simulation(
    store_step=1,
    write_step=None,
)
def darcy_convection_rayleigh_benard_semicircle(
    Nradial: int = 100,
    cell: str = CellType.TRIANGLE,
    Ra: float = 5e2,
    c_eps: float = 1e-6,
    c_freq: int = 8,
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    cfl_courant: float = 0.75,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (AB2, CN),
    D_diff: FiniteDifference = CN,
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    secondary: bool = False,
):
    """
    Non-dimensionalization choosing `ℒ` as semicircle radius, 
    `𝒰` as convective speed and `𝒯` constructed from `ℒ` and `𝒰`.
    """
    r2 = lambda x: x[0]**2 + x[1]**2
    r = lambda x, sqrt=np.sqrt: sqrt(r2(x))
    radius = 1.0
    dr = radius / Nradial
    Omega = circle_sector_mesh(dr, cell, 'semicircle')(radius, 180)
    dOmega = mesh_boundary(
        Omega, 
        {
            "lower": lambda x: x[1],
            "outer": lambda x: r2(x) - radius**2,
        },
    )
    radial_noise = lambda x: c_eps * np.sin(c_freq * np.pi * (r(x, np.sqrt)) / radius)
    c_ics = SpatialPerturbation(
        0.0,
        radial_noise,
        Omega.geometry.x,
        c_eps,
        ) 
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], 1.0),
        ("dirichlet", dOmega['outer'], 0.0),  
    )
    dispersion = lambda phi: (1/Ra) * phi
    density = lambda c: -c
    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        Ad=1,
        Pe=Ra, 
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        dispersion=dispersion,
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