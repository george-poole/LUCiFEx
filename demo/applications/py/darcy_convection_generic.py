from typing import Callable, TypeAlias, Iterable
from types import EllipsisType

import numpy as np
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr

from lucifex.fdm import FiniteDifference
from lucifex.fem import Function as Function, SpatialConstant as Constant
from lucifex.mesh import MeshBoundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, AB1, Series, 
    ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, bvp, ibvp, evaluation, 
    ds_solver, interpolation
)
from lucifex.utils import extremum
from lucifex.sim import Simulation

from lucifex.pde.streamfunction import streamfunction_velocity
from lucifex.pde.transport import advection_diffusion, flux
from lucifex.pde.darcy import darcy_streamfunction


Phi: TypeAlias = Function
C: TypeAlias = FunctionSeries
U: TypeAlias = FunctionSeries
def darcy_convection_generic(
    # domain
    Omega: Mesh,
    dOmega: MeshBoundary,
    # gravity
    egx: Expr | Function | Constant | float = 0,
    egy: Expr | Function | Constant | float = -1,
    # initial conditions
    c_ics = None,
    # boundary conditions
    c_bcs: BoundaryConditions | EllipsisType | None = Ellipsis,
    psi_bcs: BoundaryConditions | EllipsisType | None = Ellipsis,
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
    # TODO supg stabilization
    # c_stabilization: str | None = None,
    # c_limits: tuple[float, float] | EllipsisType | None = None,
    # linear algebra
    psi_petsc: OptionsPETSc | None = None,
    c_petsc: OptionsPETSc | None = None,
    # optional solvers
    secondary: bool = False,    
    # solvers
    secondary_extras: Iterable = (),
    namespace_extras: Iterable = (),
      
) -> Simulation:
    """
    2D streamfunction formulation with boundary conditions `ψ = 0` on `∂Ω`.

    Default gravity unit vector is `e₉ = -eʸ`.
    
    Default boundary conditions are no flux of solute everywhere on `∂Ω`. 
    
    Default constitutive relations are uniform porosity `ϕ = 1`, 
    isotropic quadratic permeability `K(ϕ) = ϕ²`, isotropic linear solutal
    dispersion `D(ϕ) = ϕ` and uniform viscosity `μ = 1`.
    """ 
    # time
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(Omega, "t", order, ics=0.0)  
    dt = ConstantSeries(Omega, 'dt')

    # default boundary conditions
    c_bcs = BoundaryConditions(("neumann", dOmega.union, 0.0)) if c_bcs is Ellipsis else c_bcs
    psi_bcs = BoundaryConditions(("dirichlet", dOmega.union, 0.0)) if psi_bcs is Ellipsis else psi_bcs

    # flow
    psi_deg = 2
    psi = FunctionSeries((Omega, 'P', psi_deg), 'psi')
    u = FunctionSeries((Omega, "P", psi_deg - 1, 2), "u", order)
    # transport
    c = FunctionSeries((Omega, 'P', 1), 'c', order, ics=c_ics)
    # constitutive
    phi = Function((Omega, 'P', 1), porosity, 'phi')
    k: Expr = permeability(phi)
    try:
        d = ExprSeries(dispersion(phi, u), 'd')
    except TypeError:
        d: Expr = dispersion(phi)
    rho = ExprSeries(density(c), 'rho')
    mu = ExprSeries(viscosity(c), 'mu')

    # solvers
    psi_solver = bvp(darcy_streamfunction, psi_bcs, psi_petsc)(
        psi, k, mu[0], egx * rho[0], egy * rho[0],
    )
    u_solver = interpolation(u, streamfunction_velocity)(psi[0])
    dt_solver = evaluation(dt, cfl_timestep)(
            u[0], cfl_h, cfl_courant, dt_max, dt_min,
        ) 
    c_solver = ibvp(advection_diffusion, bcs=c_bcs, petsc=c_petsc)(
        c, dt, u, d, D_adv, D_diff, phi=phi,
    )
    solvers = [psi_solver, u_solver, dt_solver, c_solver]

    if secondary:
        solvers.extend(
            [
                evaluation(ConstantSeries(Omega, "uMinMax", shape=(2,)), extremum)(u[0]),
                evaluation(ConstantSeries(Omega, "cMinMax", shape=(2,)), extremum)(c[0]),
                evaluation(ConstantSeries(Omega, "dtCFL"), cfl_timestep)(u[0], cfl_h),
                ds_solver(ConstantSeries(Omega, "fOmega", shape=(len(dOmega.union), 2)))(flux, dOmega.union)(c[0], u[0], d[0]),
            ]
        )

    namespace = [phi, ('k', k), ('d', d), rho, mu]

    solvers.extend(secondary_extras)
    namespace.extend(namespace_extras)

    return Simulation(solvers, t, dt, namespace)