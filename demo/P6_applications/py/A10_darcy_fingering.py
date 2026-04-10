from typing import Callable, TypeAlias

import numpy as np
import scipy.special as sp
from ufl import SpatialCoordinate, sqrt
from ufl.core.expr import Expr
from dolfinx.fem import FunctionSpace

from lucifex.mesh import rectangle_mesh, mesh_boundary, annulus_mesh
from lucifex.fdm import (
    FiniteDifference, FiniteDifferenceArgwise, AB2, CN, 
    advective_timestep, finite_difference_order, FunctionSeries, ConstantSeries,
)
from lucifex.fdm.ufl_overloads import exp
from lucifex.fem import Constant, Function, SpatialPerturbation, sinusoid_noise
from lucifex.solver import (
    BoundaryConditions, OptionsJIT, OptionsPETSc, bvp, ibvp,
    interpolation, evaluation,
)
from lucifex.sim import configure_simulation, Simulation
from lucifex.utils.fenicsx_utils import CellType, BoundaryType, limits_corrector
from lucifex.pde.darcy import darcy_pressure, darcy_velocity_from_pressure
from lucifex.pde.scaling import ScalingOptions
from lucifex.pde.advection_diffusion import dg_advection_diffusion, advection_diffusion

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from P5_convection.py.C10_darcy_convection_generic import darcy_convection_generic


DARCY_FINGERING_SCALINGS = ScalingOptions(
    ('Ad', 'Di', 'In', 'X'),
    lambda Pe: {
        'advective': (1, 1/Pe, 1, 1),
        'diffusive': (1, 1, Pe, 1),
        'pressure_driven': (1, 1/Pe, 0, 1), 
    }
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
    jit=OptionsJIT(Ellipsis),
)
def darcy_fingering_rectangle(
    # domain
    aspect: float = 2.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    # physical
    scaling: str = 'advective',
    Pe: float = 5e2,
    Lmbda: float = 1e1,
    left_to_right: bool = True,
    erf_eps: float = 1e-2,
    c_ampl: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_limits: tuple[float, float] | bool = False,
    bc_type: BoundaryType = BoundaryType.NEUMANN,
    # timestep
    dt_max: float = 0.25,
    dt_h: str | float = "hmin",
    dt_courant: float = 0.25,
    # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
):
    """
    `Ω = [0, A·X] × [0, X]` \\
    `∂c/∂t + 𝐮·∇c = Di ∇²c` \\
    `∇⋅𝐮 = 0` \\
    `𝐮 = -(1 / μ(c)) · (∇p + In μ(c)𝐞ˣ)`
    """
    # space
    scaling_map = DARCY_FINGERING_SCALINGS[scaling](Pe)
    X = scaling_map['X']
    Lx = aspect * X
    Ly = 1.0 * X
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
    Di, In = scaling_map(Omega)['Di', 'In']
    Pe = Constant(Omega, Pe, 'Pe')
    Lmbda = Constant(Omega, Lmbda, 'Lmbda')
    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 0.5 * (1.0 + sp.erf(-x[0] / (Lx * erf_eps))),
        sinusoid_noise(['dirichlet', bc_type], [Lx, Ly], c_freq),
        [Lx, Ly],
        c_ampl,
    ) 
    if bc_type == BoundaryType.NEUMANN:
        psi_bcs = BoundaryConditions(
            (BoundaryType.DIRICHLET, dOmega.union, 0.0),
        )
        c_bcs = BoundaryConditions(
            (BoundaryType.NEUMANN, dOmega['upper', 'lower'], 0.0),
            ("dirichlet", dOmega['left'], 1.0),
            ("dirichlet", dOmega['right'], 0.0),
        )
    elif bc_type == BoundaryType.PERIODIC:
        upper_to_lower = lambda x: np.vstack((x[0], x[1] - Ly))
        psi_bcs = BoundaryConditions(
            (BoundaryType.PERIODIC, dOmega['upper'], upper_to_lower),
            ("dirichlet", dOmega['left', 'right'], 0.0),
        )
        c_bcs = BoundaryConditions(
            (BoundaryType.PERIODIC, dOmega['upper'], upper_to_lower),
            ("dirichlet", dOmega['left'], 1.0),
            ("dirichlet", dOmega['right'], 0.0),
        )
    else:
        raise ValueError(
            f"Expected boundary type {BoundaryType.NEUMANN}' or '{BoundaryType.PERIODIC}'."
        )
    # constitutive
    dispersion = lambda phi: Di * phi
    viscosity = lambda c: exp(-Lmbda * c)
    density = lambda c: In * viscosity(c)
    if left_to_right:
        eIn = 1
    else:
        eIn = -1

    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        egx=-eIn,
        egy=0,
        dispersion=dispersion,
        density=density,
        viscosity=viscosity,
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        psi_bcs=psi_bcs,
        dt_max=dt_max, 
        dt_h=dt_h, 
        dt_courant=dt_courant,
        D_adv=D_adv,
        D_diff=D_diff,
        c_limits=c_limits,
        auxiliary=(Pe, Lmbda, Di, In),
    )


Phi: TypeAlias = Function
@configure_simulation(
    store_delta=1,
    write_delta=None,
    jit=OptionsJIT(Ellipsis),
)
def darcy_fingering_annulus(
    # domain
    Rratio: float = 0.5,
    Nradial: int = 32,
    cell: str = CellType.TRIANGLE,
    # physical
    Pe: float = 5e2,
    Lmbda: float = 1e1,
    # initial front
    zeta0_ratio: float = 0.5,
    zeta0_eps: float = 1e-2,
    c_ampl: float = 1e-6,
    c_freq: int = 8,
    # constitutive relations
    porosity: Callable[[np.ndarray], np.ndarray] | float = 1,
    permeability: Callable[[Phi], Expr] = lambda phi: phi**2,
    # timestep
    dt_min: float = 0.0,
    dt_max: float = 0.5,
    dt_h: str | float = "hmin",
    dt_courant: float | None = 0.75,
    # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
    # linear algebra
    p_petsc: OptionsPETSc = OptionsPETSc('cg', 'hypre'),
    c_petsc: OptionsPETSc = OptionsPETSc('gmres', 'ilu'),
    c_limits: tuple[float, float] | bool = False,
    # function spaces
    c_dg: float | None = 10.0,
    c_fs: FunctionSpace | None = None,
    p_fs: FunctionSpace | None = None,
    bc_type: BoundaryType = BoundaryType.NEUMANN,
):
    # space
    scaling_map = DARCY_FINGERING_SCALINGS['pressure_driven'](Pe)
    X = scaling_map['X']
    Router = 1.0 * X
    Rinner = Rratio * X
    Rfront = Rinner + zeta0_ratio * (Router - Rinner)
    r2 = lambda x: x[0]**2 + x[1]**2
    r = lambda x, sqrt=np.sqrt: sqrt(r2(x))
    dr = (Router - Rinner) / Nradial
    Omega = annulus_mesh(dr, cell)(Rinner, Router)
    dOmega = mesh_boundary(
        Omega, 
        {
            "inner": lambda x: r2(x) - Rinner**2,
            "outer": lambda x: r2(x) - Router**2,
        },
    )
    # time
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(Omega, "t", order, ics=0.0) 
    dt = ConstantSeries(Omega, 'dt')
    # constants
    Di = scaling_map(Omega)['Di']
    Pe = Constant(Omega, Pe, 'Pe')
    Lmbda = Constant(Omega, Lmbda, 'Lmbda')
    # initial and boundary conditions
    c_noise = lambda x: (
        c_ampl * np.sin(c_freq * np.pi * (r(x, np.sqrt) - Rinner) / (Router - Rinner))
    )
    c_ics = SpatialPerturbation(
        lambda x: 0.5 * (1.0 + sp.erf(-(r(x, np.sqrt) - Rfront) / (X * zeta0_eps))),
        c_noise,
        Omega.geometry.x,
        c_ampl,
    )  
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['inner'], 1.0),
        (bc_type, dOmega['outer'], 0.0),
    )
    p_bcs = BoundaryConditions(
        ("dirichlet", dOmega['inner'], 1.0),
        ("dirichlet", dOmega['outer'], 0.0),
    )
    # flow
    p = FunctionSeries(
        (Omega, 'P', 2) if p_fs is None else p_fs, 
        'p',
    )
    p_deg = p.function_space.ufl_element().degree()
    u = FunctionSeries((Omega, "P", p_deg - 1, 2), "u", order)
    # transport
    c = FunctionSeries(
        (Omega, 'DP' if c_dg else 'P', 1) if c_fs is None else c_fs, 
        'c', 
        order, 
        ics=c_ics,
    )
    # constitutive
    mu = 1 - Lmbda * c
    phi = Function((Omega, 'P', 1), porosity, 'phi')
    k = permeability(phi)
    d = Di * phi
    # solvers
    p_solver = bvp(darcy_pressure, p_bcs, p_petsc)(
        p, k, mu[0], 
    )
    u_solver = interpolation(u, darcy_velocity_from_pressure)(
        p[0], k, mu[0],
    )
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], dt_h, dt_courant, dt_max, dt_min,
    ) 
    c_limits = (0, 1) if c_limits is True else c_limits
    c_corrector = limits_corrector(*c_limits) if c_limits else None
    if c_dg:
        c_solver = ibvp(dg_advection_diffusion, petsc=c_petsc, corrector=c_corrector)(
            c, dt, c_dg, u, d, D_adv, D_diff, phi=phi, bcs=c_bcs,
        )
    else:
        c_solver = ibvp(advection_diffusion, bcs=c_bcs, petsc=c_petsc, corrector=c_corrector)(
            c, dt, u, d, D_adv, D_diff, phi=phi,
        )
    solvers = [
        p_solver, 
        u_solver, 
        dt_solver, 
        c_solver,
    ]
    auxiliary = [phi, ('k', k), ('d', d), mu, Di]
    return Simulation(solvers, t, dt, auxiliary)