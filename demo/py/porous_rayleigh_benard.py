import numpy as np
from ufl import SpatialCoordinate, sqrt

from lucifex.mesh import annulus_mesh, circle_sector_mesh, mesh_boundary
from lucifex.fdm import FiniteDifference, AB2, CN, ConstantSeries
from lucifex.solver import BoundaryConditions, OptionsPETSc, dS_solver
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, SpatialPerturbation, cubic_noise

from .porous import porous_convection_simulation, flux
from .utils import rectangle_domain, flux


@configure_simulation(
    store_step=1,
    write_step=None,
)
def porous_rayleigh_benard_rectangle(
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
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = (AB2, CN),
    D_diff: FiniteDifference = CN,
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    secondary: bool = False,
):
    Omega, dOmega = rectangle_domain(Lx, Ly, Nx, Ny, cell)
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
    density = lambda c: -c
    simulation = porous_convection_simulation(
        Omega=Omega, 
        dOmega=dOmega, 
        Ra=Ra, 
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
    if secondary:
        c, u, d = simulation['c', 'u', 'd']
        f = ConstantSeries(Omega, "f", shape=(2, ))
        simulation.solvers.append(
            dS_solver(f, flux, lambda x: x[1] - Ly / 2, facet_side="+")(c[0], u[0], d[0], Ra),
        )
    return simulation


@configure_simulation(
    store_step=1,
    write_step=None,
)
def porous_rayleigh_benard_annulus(
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
    density = lambda c: -c
    x = SpatialCoordinate(Omega)
    egx = -x[0] / r(x, sqrt)
    egy = -x[1] / r(x, sqrt)
    simulation = porous_convection_simulation(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=egx,
        egy=egy,
        Ra=Ra, 
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
    return simulation


@configure_simulation(
    store_step=1,
    write_step=None,
)
def porous_rayleigh_benard_semicircle(
    radius: float,
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
    r2 = lambda x: x[0]**2 + x[1]**2
    r = lambda x, sqrt=np.sqrt: sqrt(r2(x))
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
    density = lambda c: -c
    simulation = porous_convection_simulation(
        Omega=Omega, 
        dOmega=dOmega, 
        Ra=Ra, 
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
    return simulation