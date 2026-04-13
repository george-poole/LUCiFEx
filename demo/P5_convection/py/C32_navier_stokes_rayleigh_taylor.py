from ufl import as_vector

from lucifex.fem import Constant, SpatialPerturbation, cubic_noise
from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, FE, CN, BE,
    FiniteDifferenceArgwise, ExprSeries, finite_difference_order, advective_timestep,
)
from lucifex.solver import (
    BoundaryConditions, OptionsJIT, ibvp, evaluation,
)
from lucifex.sim import configure_simulation

from lucifex.pde.navier_stokes import ipcs_solvers
from lucifex.pde.constitutive import newtonian_stress
from lucifex.pde.advection_diffusion import advection_diffusion

from .C31_navier_stokes_thermosolutal import NAVIER_STOKES_CONVECTION_SCALINGS


@configure_simulation(
    store_delta=1,
    write_delta=None,
    jit=OptionsJIT(Ellipsis),
)
def navier_stokes_rayleigh_taylor_rectangle(
    # domain
    aspect: float = 2.0,
    Nx: int = 64,
    Ny: int = 64,
    cell: str = 'quadrilateral',
    # physical
    scaling: str = 'diffusive',
    Pr = 1.0,
    Ra = 1e3,
    # initial perturbation
    c_ampl: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_seed: tuple[int, int] = (12, 34),
    # timestep
    dt_max: float = 0.5,
    dt_min: float = 0.0,
    dt_Cu: float = 0.75,
    # time discretization
    D_adv_ns: FiniteDifference = FE,
    D_visc_ns: FiniteDifference = CN,
    D_buoy_ns: FiniteDifference = FE,
    D_adv_c: FiniteDifference | FiniteDifferenceArgwise = (BE @ BE),
    D_diff_c: FiniteDifference = CN,
):
    """
    `∂c/∂t + 𝐮·∇c = Di∇²c` \\
    `∇·𝐮 = 0` \\
    `∂𝐮/∂t + 𝐮·∇𝐮 = Vi(-∇p + ∇²𝐮) - Bu c 𝐞ʸ`
    """
    scaling_map = NAVIER_STOKES_CONVECTION_SCALINGS[scaling](Ra, Pr)
    X = scaling_map['X']
    Lx = aspect * X
    Ly = 1.0 * X
    # space
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
        D_adv_ns, D_visc_ns, D_buoy_ns, D_adv_c, D_diff_c,
    )
    t = ConstantSeries(Omega, 't', ics=0.0)
    dt = ConstantSeries(Omega, 'dt')
    # constants
    Di, Vi, Bu = scaling_map(Omega)['Di', 'Vi', 'Bu']
    Pr = Constant(Omega, Pr, 'Pr')
    Ra = Constant(Omega, Ra, 'Ra')  
    # initial and boundary conditions
    c_ics = SpatialPerturbation(
        lambda x: 1.0 * (x[1] > Ly / 2) + 0.0,
        cubic_noise(['neumann', 'neumann'], [Lx, Ly], c_freq, c_seed),
        [Lx, Ly],
        c_ampl,
    )   
    c_bcs = BoundaryConditions(
        ('neumann', dOmega.union, 0.0)
    )
    u_bcs = BoundaryConditions(
        ('dirichlet', dOmega.union, u_zero),
    )
    # flow and transport
    u = FunctionSeries((Omega, 'P', 2, dim), 'u', order, ics=u_zero)
    p = FunctionSeries((Omega, 'P', 1), 'p', order, ics=0.0)
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    deviatoric_stress = lambda u: Vi * newtonian_stress(u, 1)
    rho = ExprSeries(c, 'rho')
    f = Bu * rho * as_vector([0, -1]) 
    # solvers
    dt_solver = evaluation(dt, advective_timestep)(
        u[0], 'hmin', dt_Cu, dt_max, dt_min,
    )
    ns_solvers = ipcs_solvers(
        u, p, dt[0], deviatoric_stress, D_adv_ns, D_visc_ns, D_buoy_ns, 
        f=f, 
        u_bcs=u_bcs, 
        p_scale=Vi,
    )
    c_solver = ibvp(advection_diffusion, bcs=c_bcs)(
        c, dt[0], u, Di, D_adv_c, D_diff_c,
    )
    solvers = [dt_solver, *ns_solvers, c_solver]
    auxiliary = [Pr, Ra, Di, Vi, Bu, rho]
    return solvers, t, dt, auxiliary

