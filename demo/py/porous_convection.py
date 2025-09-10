from typing import Callable, TypeAlias
from types import EllipsisType

import numpy as np
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
from ufl import as_matrix

from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.mesh import MeshBoundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, AB1, Series, 
    ExprSeries, finite_difference_order, cfl_timestep,
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, eval_solver, 
    ds_solver, dS_solver
)
from lucifex.utils import extremum
from lucifex.sim import Simulation


def flux():
    ...


def darcy_streamfunction():
    ...


def streamfunction_velocity(psi: Function) -> Expr:
    """
    `𝐮 = ((0, 1), (-1, 0))·∇ψ`
    """
    return as_matrix([[0, 1], [-1, 0]]) * grad(psi)


def advection_diffusion():
    ...


C: TypeAlias = FunctionSeries
Phi: TypeAlias = FunctionSeries
U: TypeAlias = FunctionSeries
def abstract_porous_convection(
    # domain
    Omega: Mesh,
    dOmega: MeshBoundary,
    # physical 
    Ra: float = 0,
    # initial conditions
    c_ics = None,
    # boundary conditions
    c_bcs: BoundaryConditions | EllipsisType | None = None,
    # constitutive relations
    phi: Callable[[np.ndarray], np.ndarray] | float = 1,
    permeability: Callable[[Phi], Series] = lambda phi: phi**2,
    dispersion: Callable[[Phi, U], Series] = lambda phi, _: phi,
    density: Callable[[C], Series] = lambda c: c,
    viscosity: Callable[[C], Series] = lambda *_: 1 + 0 * _[0], 
    # time step
    dt_min: float = 0.0,
    dt_max: float = 0.5,
    cfl_h: str | float = "hmin",
    courant: float | None = 0.75,
    # time discretization
    D_adv: FiniteDifference | tuple[FiniteDifference, FiniteDifference] = AB1,
    D_diff: FiniteDifference = AB1,
    D_reac: FiniteDifference 
    | tuple[FiniteDifference, FiniteDifference]
    | tuple[FiniteDifference, FiniteDifference, FiniteDifference] = AB1,
    # linear algebra
    flow_petsc: tuple[OptionsPETSc | None, OptionsPETSc | EllipsisType | None] = (None, ...),
    c_petsc: OptionsPETSc | None = None,
    # optional solvers
    secondary: bool = False,      
) -> Simulation:    

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
    phi = Function((Omega, 'P', 1), phi, 'phi')
    k = ExprSeries(permeability(phi), 'k')
    d = ExprSeries(dispersion(phi, u), 'd')
    rho = ExprSeries(density(c), 'rho')
    mu = ExprSeries(viscosity(c), 'mu')
    
    namespace = [Ra, phi, k, d, rho, mu]

    # flow solvers
    psi_bcs = BoundaryConditions(("dirichlet", dOmega.union, 0.0))
    psi_solver = ...
    u_solver = ...

    # timestep solver
    dt_solver = eval_solver(dt, cfl_timestep)(
            u[0], cfl_h, courant, dt_max, dt_min,
        ) 

    # transport solvers
    c_bcs = BoundaryConditions(("neumann", dOmega.union, 0.0)) if c_bcs is Ellipsis else c_bcs
    c_limits = (0, 1) if c_limits is Ellipsis else c_limits
    c_solver = ...

    solvers = [psi_solver, u_solver, dt_solver, c_solver]

    # optional solvers
    if secondary:
        solvers.extend(
            [
                eval_solver(ConstantSeries(Omega, "uMinMax", shape=(2,)), extremum)(u[0]),
                eval_solver(ConstantSeries(Omega, "cMinMax", shape=(2,)), extremum)(c[0]),
                eval_solver(ConstantSeries(Omega, "dtCFL"), cfl_timestep)(u[0], cfl_h),
                ds_solver(ConstantSeries(Omega, "f", shape=(len(dOmega.union), 2)))(flux, dOmega.union)(c[0], u[0], d[0], Ra),
            ]
        )

    
    return Simulation(solvers, t, dt, namespace)