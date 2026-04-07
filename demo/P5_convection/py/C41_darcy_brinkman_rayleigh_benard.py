from typing import Callable

import numpy as np
from ufl import as_vector, sqrt

from lucifex.fdm import FiniteDifference, FE, CN, BE
from lucifex.fem import Function, Constant, SpatialPerturbation, cubic_noise
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, FiniteDifferenceArgwise,
    ExprSeries, finite_difference_order, advective_timestep,
)
from lucifex.solver import (
    BoundaryConditions, OptionsJIT, ibvp, evaluation,
)
from lucifex.fem import SpatialPerturbation, cubic_noise
from lucifex.sim import configure_simulation

from lucifex.pde.navier_stokes import ipcs_solvers
from lucifex.pde.constitutive import newtonian_stress
from lucifex.pde.advection_diffusion import advection_diffusion
from lucifex.pde.scaling import ScalingOptions


DARCY_BRINKMAN_CONVECTION_SCALINGS = ScalingOptions(
    ('Ad', 'Di', 'Vi', 'Bu', 'Pm', 'X'),
    lambda Ra, Pr, Dr: {
        'advective': (1, 1/np.sqrt(Ra*Pr), np.sqrt(Pr/Ra), 1, np.sqrt(Pr/Ra)/Dr, 1),
    }
)


@configure_simulation(
    store_delta=1,
    write_delta=None,
    jit=OptionsJIT(Ellipsis),
)
def darcy_brinkman_rayleigh_benard_rectangle(    
    # domain
    aspect: float = 2.0,
    Nx: int = 64,
    Ny: int = 64,
    cell: str = 'quadrilateral',
    # physical
    scaling: str = 'advective',
    Pr: float = 1.0,
    Ra: float = 1e7,
    Dr: float = 1e-4,
    phi: float = 1,
    mu: float = 1,
    inverse_permeability: Callable[[np.ndarray], np.ndarray] | None = None,
    # initial perturbation
    noise_eps: float = 1e-6,
    noise_freq: tuple[int, int] = (8, 8),
    noise_seed: tuple[int, int] = (12, 34),
    # timestep
    dt_max: float = 0.5,
    dt_min: float = 0.0,
    dt_courant: float = 0.5,
    # time discretization
    D_adv_ns: FiniteDifference = FE,
    D_visc_ns: FiniteDifference = CN,
    D_buoy_ns: FiniteDifference = FE,
    D_darcy_ns: FiniteDifference = BE,
    D_adv_ad: FiniteDifference | FiniteDifferenceArgwise = BE,
    D_diff_ad: FiniteDifference = CN,
):
    """
    `∂c/∂t + Ad 𝐮·∇c = Di ∇²c` \\
    `∇⋅𝐮 = 0` \\
    `∂𝐮/∂t + 𝐮·∇𝐮 = -∇p + Vi ∇²𝐮 + Bu c𝐞ʸ - Pm K⁻¹·𝐮`
    """
    # space
    scaling_map = DARCY_BRINKMAN_CONVECTION_SCALINGS[scaling](Ra, Pr, Dr)
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
    u_zero = [0.0] * dim
    # time
    order = finite_difference_order(
        D_adv_ns, D_visc_ns, D_buoy_ns, D_adv_ad, D_diff_ad,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')
    # constants
    Ad, Di, Vi, Bu, Pm = scaling_map[Omega, 'Ad', 'Di', 'Vi', 'Bu', 'Pm']
    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')
    Dr = Constant(Omega, Dr, 'Da')
    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 1 - x[1],
        cubic_noise(['dirichlet', 'dirichlet'], [Lx, Ly], noise_freq, noise_seed),
        [Lx, Ly],
        noise_eps,
    )
    c_bcs = BoundaryConditions(
        ('dirichlet', dOmega['lower'], 1.0),
        ('dirichlet', dOmega['upper'], 0.0),
        ('neumann', dOmega['left', 'right'], 0.0)
    )  
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, u_zero),
    )  
    # flow
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    # transport
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    if inverse_permeability is None:
        inverse_permeability = lambda x: 1.0 - 1.0 * (x[1] > Ly / 2)
    kInv = Function(
        (Omega, 'DP', 0), 
        inverse_permeability,
        name='kInv',
    )
    rho = ExprSeries(-c, 'rho')
    eg = as_vector([0, -1])
    f = (
        Bu * phi * D_buoy_ns(rho, trial=u) * eg 
        - Pm * mu * phi * kInv * D_darcy_ns(u, trial=u)
    )
    stress = lambda u: Vi * newtonian_stress(u, mu) # FIXME efffect of phi on stress and bcs
    # solvers
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], 'hmin', dt_courant, dt_max, dt_min,
    )
    ns_solvers = ipcs_solvers(
        u, p, dt[0], stress, D_adv_ns, D_visc_ns, 
        f=f, 
        u_bcs=u_bcs,
        adv_scale=Ad, 
        p_scale=phi, # FIXME efffect of phi on stress and bcs
    )
    c_solver = ibvp(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, Di, D_adv_ad, D_diff_ad,
    )
    solvers = (dt_solver, *ns_solvers, c_solver)
    auxiliary = (
        rho, kInv,
        Pr, Ra, Dr,
        Ad, Di, Vi, Bu, Pm,
    )
    return solvers, t, dt, auxiliary