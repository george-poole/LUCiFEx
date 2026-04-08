from typing import Callable, TypeAlias

import numpy as np
import scipy.special as sp

from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, FiniteDifference, FiniteDifferenceArgwise, AB1, Series, ConstantSeries, 
    finite_difference_order, ExprSeries, adr_timestep,
)
from lucifex.fem import Constant,  SpatialPerturbation, cubic_noise
from lucifex.solver import(
    BoundaryConditions, OptionsPETSc, OptionsJIT, 
    interpolation, ibvp, bvp, evaluation,
)
from lucifex.sim import configure_simulation
from lucifex.utils.fenicsx_utils import CellType, limits_corrector

from lucifex.pde.advection_diffusion import advection_diffusion_reaction
from lucifex.pde.darcy import darcy_streamfunction
from lucifex.pde.streamfunction_vorticity import velocity_from_streamfunction
from lucifex.pde.scaling import ScalingOptions


DARCY_ABC_SCALINGS = ScalingOptions(
    ('Ad', 'Di', 'Ki', 'Bu', 'X'),
    lambda Ra, Da: {
        'advective': (1, 1/Ra, Da, 1, 1),
        'diffusive': (1, 1, Ra * Da, Ra, 1),
        'advective_diffusive': (1, 1, Da/Ra, 1, Ra),
        'reactive': (1, 1, 1, np.sqrt(Ra / Da), np.sqrt(Ra * Da)),
    }
)


A: TypeAlias = FunctionSeries
B: TypeAlias = FunctionSeries
C: TypeAlias = FunctionSeries
@configure_simulation(
    store_delta=1,
    write_delta=None,
    jit=OptionsJIT(Ellipsis),
)
def darcy_abc_convection_rectangle(
    # domain
    aspect: float = 2.0, 
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    # physical
    scaling: str = 'reactive',
    Ra: float = 5e2,
    Da: float = 1e2,
    Le_b: float = 1.0,
    Le_c: float = 1.0,
    Lr_b: float = 1.0,
    Lr_c: float = 1.0,
    gamma_a: float = 1.0,
    gamma_b: float = 1.0,
    gamma_c: float = 1.0,
    reaction: Callable[[A, B, C], Series] = lambda a, b, _: a * b,
    # initial conditions
    a_ampl: float = 1e-6,
    a_freq: tuple[int, int] = (8, 8),
    a_seed: tuple[int, int] = (1234, 5678),
    a_eps: float = 1e-2,
    b0: float = 1.0,
    c0: float = 0.0,
    # timestep
    dt_min: float = 0.0,
    dt_max: float = np.inf,
    dt_h: str | float = "hmin",
    courant_adv: float | None = 1.0,
    courant_diff: float | None = 1.0,
    courant_reac: float | None = 1.0,
    # time discretization
    D_adv: FiniteDifference | FiniteDifferenceArgwise = AB1,
    D_diff: FiniteDifference = AB1,
    D_src_a: FiniteDifference = AB1,
    D_src_b: FiniteDifference = AB1,
    D_src_c: FiniteDifference = AB1,
    # linear algebra
    psi_petsc: OptionsPETSc = OptionsPETSc('cg', 'hypre'),
    abc_petsc: OptionsPETSc = OptionsPETSc('gmres', 'ilu'),
    a_limits: tuple[float | None, float | None] | None = None,
    b_limits: tuple[float | None, float | None] | None = None,
    c_limits: tuple[float | None, float | None] | None = None,
):
    # space
    scaling_map = DARCY_ABC_SCALINGS[scaling](Ra, Da)
    X = scaling_map['X']
    Lx = aspect * X
    Ly = 1.0 * X
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
    # time
    order = finite_difference_order(
        D_adv, D_diff, D_src_a, D_src_b, D_src_c,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')
    # constants
    Di, Ki, Bu = scaling_map[Omega, 'Di', 'Ki', 'Bu']
    Ra = Constant(Omega, Ra, 'Ra')
    Da = Constant(Omega, Da, 'Da')
    Le_b = Constant(Omega, Le_b, 'Leb')
    Le_c = Constant(Omega, Le_c, 'Lec')
    Lr_b = Constant(Omega, Le_b, 'Lrb')
    Lr_c = Constant(Omega, Le_c, 'Lrc')
    gamma_a = Constant(Omega, gamma_a, 'gamma_a')
    gamma_b = Constant(Omega, gamma_b, 'gamma_b')
    gamma_c = Constant(Omega, gamma_c, 'gamma_c')
    # initial and boundary conditions
    a_ics = SpatialPerturbation(
        lambda x: 1 + sp.erf((x[1] - Ly) / (X * a_eps)),
        cubic_noise(['neumann', ('neumann', 'dirichlet')], [Lx, Ly], a_freq, a_seed),
        [Lx, Ly],
        a_ampl,
    ) 
    a_bcs = BoundaryConditions(
        ('dirichlet', dOmega['upper'], 1.0),
        ('neumann', dOmega['left', 'right', 'lower'], 0.0),
    )
    b_bcs = BoundaryConditions(
        ('neumann', dOmega.union, 0.0),
    )
    c_bcs = BoundaryConditions(
        ('neumann', dOmega.union, 0.0),
    )
    psi_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, 0.0),
    )
    # flow
    psi_deg = 2
    psi = FunctionSeries((Omega, 'P', psi_deg), 'psi')
    u = FunctionSeries((Omega, "P", psi_deg - 1, 2), "u", order)
    # transport
    a = FunctionSeries((Omega, 'P', 1), 'a', order, ics=a_ics)
    b = FunctionSeries((Omega, 'P', 1), 'b', order, ics=b0)
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c0)
    # constitutive
    rho = ExprSeries(gamma_a * a + gamma_b * b + gamma_c * c, 'rho')
    Sigma = ExprSeries(reaction(a, b, c), 'Sigma')
    mu = 1
    phi = 1
    k = 1
    # solvers
    psi_solver = bvp(darcy_streamfunction, psi_bcs, psi_petsc)(
        psi, k, mu, fy=-Bu * rho[0],
    )
    u_solver = interpolation(u, velocity_from_streamfunction)(psi[0])
    dt_solver = evaluation(dt, adr_timestep)(
            u[0], Di, Sigma[0], dt_h, courant_adv, courant_diff, courant_reac, dt_max, dt_min,
        ) 
    a_corrector = limits_corrector(*a_limits) if a_limits else None
    a_solver = ibvp(advection_diffusion_reaction, bcs=a_bcs, petsc=abc_petsc, corrector=a_corrector)(
        a, dt, u, Di, j=-Ki* Sigma, 
        D_adv=D_adv, D_diff=D_diff, D_src=D_src_a, phi=phi,
    )
    b_corrector = limits_corrector(*b_limits) if c_limits else None
    b_solver = ibvp(advection_diffusion_reaction, bcs=b_bcs, petsc=abc_petsc, corrector=b_corrector)(
        b, dt, u, Le_b * Di, j=-Lr_b * Ki* Sigma, 
        D_adv=D_adv, D_diff=D_diff, D_src=D_src_b, phi=phi,
    )
    c_limits = (0, 1) if c_limits is True else c_limits
    c_corrector = limits_corrector(*c_limits) if c_limits else None
    c_solver = ibvp(advection_diffusion_reaction, bcs=c_bcs, petsc=abc_petsc, corrector=c_corrector)(
        c, dt, u, Le_c * Di, j=Lr_c * Ki * Sigma, 
        D_adv=D_adv, D_diff=D_diff, D_src=D_src_c, phi=phi,
    )
    solvers = (psi_solver, u_solver, dt_solver, a_solver, b_solver, c_solver)
    auxiliary = (
        Di, Ki, Bu,
        Ra, Da, 
        Le_b, Le_c,
        Lr_b, Lr_b,
        rho, Sigma,
    )
    return solvers, t, dt, auxiliary