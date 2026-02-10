from typing import Callable

from ufl import as_vector

from lucifex.fdm import FiniteDifference, AB2, CN
from lucifex.fdm.ufl_operators import exp
from lucifex.fem import Constant, SpatialPerturbation, cubic_noise
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, Series, 
    ExprSeries, FiniteDifferenceArgwise, finite_difference_order, advective_timestep,
)
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, bvp, ibvp, evaluation, 
)
from lucifex.sim import configure_simulation
from lucifex.utils import CellType, limits_corrector

from lucifex.pde.advection_diffusion import advection_diffusion
from lucifex.pde.stokes import stokes_incompressible
from lucifex.pde.scaling import ScalingChoice
from lucifex.pde.constitutive import newtonian_stress


STOKES_CONVECTION_SCALINGS = ScalingChoice(
    ('Ad', 'Di', 'Bu', 'X'),
    lambda Ra: {
        'advective': (1, 1/Ra, 1, 1),
        'diffusive': (1, 1, Ra, 1),
    }
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
    c_ampl: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_seed: tuple[int, int] = (1234, 5678),
    # constitutive
    Lmbda: float | None = None,
    density: Callable[[FunctionSeries], Series] = lambda c: -c, 
    # time step
    dt_min: float = 0.0,
    dt_max: float = 0.5,
    dt_courant: float = 0.75,
    # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
    c_limits: bool = False,
    # linear algebra
    up_petsc: OptionsPETSc = OptionsPETSc(pc_type='lu', pc_factor_mat_solver_type='mumps'),
    c_petsc: OptionsPETSc = OptionsPETSc('gmres', 'ilu'),
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
    Di, Bu = scaling_map[Omega, 'Di', 'Bu']
    Ra = Constant(Omega, Ra, 'Ra')
    Lmbda = Constant(Omega, Lmbda, 'Lmbda')
    u_zero = [0.0] * dim
    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 1 - x[1] / Ly,
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], c_freq, c_seed),
        [Lx, Ly],
        c_ampl,
    )   
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], 1.0),
        ("dirichlet", dOmega['upper'], 0.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )
    u_bcs = BoundaryConditions(
        ('essential', dOmega.union, u_zero, 0),
    )  
    # flow and transport
    up = FunctionSeries((Omega, [('P', 2, 2,), ('P', 1)]), ('up', ['u', 'p']), order)
    u, p = up.split()
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    if Lmbda:
        viscosity = lambda c: exp(-Lmbda * c)
    else:
        viscosity = lambda c: 1 + 0 * c
    mu = ExprSeries(
        viscosity(c), 
        name='mu',
    )
    rho = ExprSeries(density(c), 'rho')
    stress = lambda u: newtonian_stress(u, mu[0])
    eg = as_vector([0, -1])
    f = Bu * rho * eg
    # solvers
    up_solver = bvp(stokes_incompressible, u_bcs, petsc=up_petsc)(up, stress, f[0])
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], 'hmin', dt_courant, dt_max, dt_min,
    )
    c_corrector = ('cCorr', limits_corrector(0, 1)) if c_limits else None
    c_solver = ibvp(advection_diffusion, bcs=c_bcs, petsc=c_petsc, corrector=c_corrector)(
        c, dt[0], u, Di, D_adv, D_diff,
    )
    solvers = [up_solver, dt_solver, c_solver]
    exprs_consts = [Ra, Lmbda, Di, Bu, u, p, mu, rho]
    return solvers, t, dt, exprs_consts