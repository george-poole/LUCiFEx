from typing import Callable, TypeAlias
from types import EllipsisType

import numpy as np
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
from ufl import (dx, Form, FacetNormal, inner,
                 as_matrix, Dx, TrialFunction, TestFunction,
                 det, transpose,  as_matrix)

from lucifex.fdm import DT, FiniteDifference
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.mesh import MeshBoundary, rectangle_mesh, mesh_boundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, AB1, Series, 
    ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, bvp_solver, ibvp_solver, eval_solver, 
    ds_solver, interpolation_solver
)
from lucifex.utils import extremum, is_tensor
from lucifex.sim import Simulation


def rectangle_domain(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str,
    name: str = 'LxLy',
    clockwise_names: tuple[str, str, str, str] = ('upper', 'right', 'lower', 'left'),
) -> tuple[Mesh, MeshBoundary]:
    
    mesh = rectangle_mesh(Lx, Ly, Nx, Ny, name, cell)
    boundary = mesh_boundary(
        mesh,
        {
            clockwise_names[0]: lambda x: x[1] - Ly,
            clockwise_names[1]: lambda x: x[0] - Lx,
            clockwise_names[2]: lambda x: x[1],
            clockwise_names[3]: lambda x: x[0],
        },
    )

    return mesh, boundary


def flux(
    c: Function,
    u: Function | Constant,
    d: Function,
    Ra: Constant
) -> tuple[Expr, Expr]:
    flux_adv = inner(u, grad(c))
    n = FacetNormal(c.function_space.mesh)
    flux_diff =  (1/Ra) * inner(n, d * grad(c))
    return flux_adv, flux_diff


def darcy_streamfunction(
    psi: FunctionSeries,
    k: Expr | Function | Constant | float,
    mu: Expr | Function | Constant | float,
    rho: Expr | Function,
    egx: Expr | Function | Constant | float | None = None,
    egy: Expr | Function | Constant | float | None = None,
) -> tuple[Form, Form]:
    v = TestFunction(psi.function_space)
    psi_trial = TrialFunction(psi.function_space)
    if is_tensor(k):
        F_lhs = -(mu / det(k)) * inner(grad(v), transpose(k) * grad(psi_trial)) * dx 
    else:
        F_lhs = -(mu / k) * inner(grad(v), grad(psi_trial)) * dx
    forms = [F_lhs]
    if egx is not None:
        F_egx = -v * Dx(egx * rho, 1) * dx
        forms.append(F_egx)
    if egy is not None:
        F_egy = v * Dx(egy * rho, 0) * dx
        forms.append(F_egy)
    return forms


def streamfunction_velocity(psi: Function) -> Expr:
    return as_matrix([[0, 1], [-1, 0]]) * grad(psi)


def porous_advection_diffusion(
    c: FunctionSeries,
    dt: Constant,
    phi: Series | Function | Expr,
    u: FunctionSeries,
    Ra: Constant,
    d: Series | Function | Expr,
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference],
    D_diff: FiniteDifference,
    D_phi: FiniteDifference = AB1,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    v = TestFunction(c.function_space)

    if isinstance(phi, Series):
        phi = D_phi(phi)
    if isinstance(d, Series):
        d = D_phi(d)

    F_dcdt = v * DT(c, dt) * dx

    match D_adv:
        case D_adv_u, D_adv_c:
            adv = (1 / phi) * inner(D_adv_u(u, False), grad(D_adv_c(c)))
        case D_adv:
            adv = (1 / phi) * D_adv(inner(u, grad(c)))
    F_adv = v * adv * dx

    F_diff = (1/Ra) * inner(grad(v / phi), d * grad(D_diff(c))) * dx

    forms = [F_dcdt, F_adv, F_diff]

    if bcs is not None:
        ds, c_neumann = bcs.boundary_data(c.function_space, 'neumann')
        F_neumann = sum([-(1 / Ra) * v * cN * ds(i) for i, cN in c_neumann])
        forms.append(F_neumann)

    return forms


Phi: TypeAlias = Function
C: TypeAlias = FunctionSeries
U: TypeAlias = FunctionSeries
def porous_convection_simulation(
    # domain
    Omega: Mesh,
    dOmega: MeshBoundary,
    # gravity
    egx: Expr | Function | Constant | float | None = None,
    egy: Expr | Function | Constant | float = -1.0,
    # physical
    Ra: float = 1000.0,
    # initial conditions
    c_ics = None,
    # boundary conditions
    c_bcs: BoundaryConditions | EllipsisType | None = None,
    # constitutive relations
    porosity: Callable[[np.ndarray], np.ndarray] | float = 1,
    permeability: Callable[[Phi], Expr] = lambda phi: phi**2,
    dispersion: Callable[[Phi, U], Series] | Callable[[Phi], Function] = lambda phi: phi,
    density: Callable[[C], Series] = lambda c: c,
    viscosity: Callable[[C], Series] = lambda c: 1 + 0 * c, 
    # time step
    dt_min: float = 0.0,
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    cfl_courant: float | None = 0.75,
    # time discretization
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = AB1,
    D_diff: FiniteDifference = AB1,
    D_reac: FiniteDifference 
    | tuple[FiniteDifference, FiniteDifference]
    | tuple[FiniteDifference, FiniteDifference, FiniteDifference] = AB1,
    # TODO supg stabilization
    # c_stabilization: str | None = None,
    # c_limits: tuple[float, float] | EllipsisType | None = None,
    # linear algebra
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    # optional solvers
    secondary: bool = False,      
) -> Simulation:
    """
    2D streamfunction formulation with boundary conditions `ψ = 0` on `∂Ω`.

    Default gravity unit vector is `e₉ = -eʸ`.
    
    Default boundary conditions are no flux of solute everywhere on `∂Ω`. 
    
    Default constitutive relations are uniform porosity `ϕ = 1`, 
    isotropic quadratic permeability `K(ϕ) = ϕ²`, isotropic linear solutal
    dispersion `D(ϕ) = ϕ` and uniform viscosity `μ = 1`.
    """    

    order = finite_difference_order(D_adv, D_diff, D_reac)

    # flow fields
    psi_deg = 2
    psi = FunctionSeries((Omega, 'P', psi_deg), 'psi')
    u = FunctionSeries((Omega, "P", psi_deg - 1, 2), "u", order)

    # transport field
    t = ConstantSeries(Omega, "t", order, ics=0.0)  
    dt = ConstantSeries(Omega, 'dt')
    Ra = Constant(Omega, Ra, 'Ra')
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)

    # constitutive relations
    phi = Function((Omega, 'P', 1), porosity, 'phi')
    k: Expr = permeability(phi)
    try:
        d = ExprSeries(dispersion(phi, u), 'd')
    except TypeError:
        d: Expr = dispersion(phi)
    rho = ExprSeries(density(c), 'rho')
    mu = ExprSeries(viscosity(c), 'mu')
    
    namespace = [Ra, phi, ('k', k), ('d', d), rho, mu]

    # flow solvers
    psi_bcs = BoundaryConditions(("dirichlet", dOmega.union, 0.0))
    psi_solver = bvp_solver(darcy_streamfunction, psi_bcs, psi_petsc)(psi, k, mu[0], rho[0], egx, egy)
    u_solver = interpolation_solver(u, streamfunction_velocity)(psi[0])

    # timestep solver
    dt_solver = eval_solver(dt, cfl_timestep)(
            u[0], cfl_h, cfl_courant, dt_max, dt_min,
        ) 

    # transport solver
    c_bcs = BoundaryConditions(("neumann", dOmega.union, 0.0)) if c_bcs is Ellipsis else c_bcs
    c_solver = ibvp_solver(porous_advection_diffusion, bcs=c_bcs, petsc=c_petsc)(
        c, dt, phi, u, Ra, d[0] if isinstance(d, ExprSeries) else d, D_adv, D_diff,
    )

    solvers = [psi_solver, u_solver, dt_solver, c_solver]

    # optional solvers
    if secondary:
        solvers.extend(
            [
                eval_solver(ConstantSeries(Omega, "uMinMax", shape=(2,)), extremum)(u[0]),
                eval_solver(ConstantSeries(Omega, "cMinMax", shape=(2,)), extremum)(c[0]),
                eval_solver(ConstantSeries(Omega, "dtCFL"), cfl_timestep)(u[0], cfl_h),
                ds_solver(ConstantSeries(Omega, "fOmega", shape=(len(dOmega.union), 2)))(flux, dOmega.union)(c[0], u[0], d[0], Ra),
            ]
        )
    
    return Simulation(solvers, t, dt, namespace)