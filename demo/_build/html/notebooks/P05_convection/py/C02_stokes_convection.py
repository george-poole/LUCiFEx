from collections.abc import Iterable
from typing import Callable, TypeAlias
from types import EllipsisType

import numpy as np
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
from ufl import inner, sqrt, as_vector

from lucifex.fdm import FiniteDifference, AB2, CN
from lucifex.fem import Function, Constant, SpatialPerturbation, cubic_noise
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, FE, Series, 
    ExprSeries, FiniteDifferenceArgwise, finite_difference_order, advective_timestep,
)
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, Solver, bvp, ibvp, evaluation, 
    integration, interpolation, extrema, L_norm,
)
from lucifex.sim import configure_simulation
from lucifex.utils import CellType

from lucifex.pde.streamfunction_vorticity import velocity_from_streamfunction
from lucifex.pde.advection_diffusion import advection_diffusion, flux
from lucifex.pde.stokes import stokes_incompressible
from lucifex.pde.scaling import ScalingOptions
from lucifex.pde.constitutive import newtonian_stress


STOKES_CONVECTION_SCALINGS = ScalingOptions(
    ('Ad', 'Di', 'Vi', 'Bu', 'X'),
    ...,
)
"""
Choice of length scale `ℒ`, velocity scale `𝒰`
and time scale `𝒯` in the non-dimensionalization.
"""


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def stokes_rayleigh_benard_rectangle(
    # domain
    aspect: float = 2.0,
    Nx: int = 64,
    Ny: int = 64,
    cell: str = CellType.QUADRILATERAL,
    # physical
    scaling: str = 'advective',
    Ra: float = 1e2,
    c_eps: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_seed: tuple[int, int] = (1234, 5678),
    # consitutive
    viscosity: Callable[[FunctionSeries], Series] = lambda c: 1 + 0 * c, 
    # time step
    dt_min: float = 0.0,
    dt_max: float = 0.5,
    cfl_courant: float = 0.75,
    # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
    # linear algebra
    up_petsc: OptionsPETSc = OptionsPETSc(pc_type='lu', pc_factor_mat_solver_type='mumps'),
    c_petsc: OptionsPETSc = OptionsPETSc('gmres', 'ilu'),
    diagnostic: bool = False,
):
    """
    `Ω = [0, A·X] × [0, X]` \\
    `∂c/∂t + 𝐮·∇c = Di ∇²c` \\
    `∇⋅𝐮 = 0` \\
    `𝟎 = -∇p + Vi ∇·𝜏(𝐮) - Bu c𝐞ʸ`

    `scaling` determines `Di, Bu, Xl` from `Ra`.
    """
    # space
    scaling_map = STOKES_CONVECTION_SCALINGS[scaling](Ra)
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
    dim = Omega.geometry.dim
    # time
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')
    # constants
    Di, Vi, Bu = scaling_map[Omega, 'Di', 'Vi', 'Bu']
    Ra = Constant(Omega, Ra, 'Ra')
    u_zero = [0.0] * dim
    # initial and boundary conditions
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
    u_bcs = BoundaryConditions(
        ('essential', dOmega.union, u_zero),
    )  
    # flow and transport
    up = FunctionSeries((Omega, [('P', 2, 2,), ('P', 1)]), ('up', ['u', 'p']), order)
    u, p = up.split()
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    mu = ExprSeries(viscosity(mu), 'mu')
    stress = lambda u: Vi * newtonian_stress(u, mu)
    rho = ExprSeries(c, 'rho')
    eg = as_vector([0, -1])
    f = -Bu * rho * eg
    # solvers
    up_solver = bvp(stokes_incompressible, u_bcs, petsc=up_petsc)(up, stress, f)
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], 'hmin', cfl_courant, dt_max, dt_min,
    )
    c_solver = ibvp(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, Di, D_adv, D_diff,
    )
    solvers = [up_solver, dt_solver, c_solver]
    exprs_consts = [Ra, Di, Vi, Bu, u, p, mu, rho]
    return solvers, t, dt, exprs_consts