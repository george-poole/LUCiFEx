from collections.abc import Iterable
from typing import Callable, TypeAlias
from types import EllipsisType

import numpy as np
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
from ufl import inner, sqrt

from lucifex.fdm import FiniteDifference
from lucifex.fem import Function, Constant
from lucifex.mesh import MeshBoundary
from lucifex.fdm import (
    FunctionSeries, ConstantSeries, FiniteDifference, FE, Series, 
    ExprSeries, FiniteDifferenceArgwise, finite_difference_order, cfl_timestep,
)
from lucifex.solver import (
    BoundaryConditions, OptionsPETSc, Solver, bvp, ibvp, evaluation, 
    integration, interpolation, extrema, L_norm,
)
from lucifex.sim import Simulation

from lucifex.pde.streamfunction import streamfunction_velocity
from lucifex.pde.advection_diffusion import advection_diffusion, flux
from lucifex.pde.darcy import darcy_streamfunction
from lucifex.pde.scaling import ScalingOptions


DARCY_CONVECTION_SCALINGS = ScalingOptions(
    ('Ad', 'Di', 'Bu', 'X'),
    lambda Ra: {
        'advective': (1, 1/Ra, 1, 1),
        'diffusive': (1, 1, Ra, 1),
        'advective_diffusive': (1, 1, 1, Ra),
    }
)
"""
Choice of length scale `ℒ`, velocity scale `𝒰`
and time scale `𝒯` in the non-dimensionalization.

`'advective'` \\
`ℒ` = domain size \\
`𝒰` = advective speed

`'diffusive'` \\
`ℒ` = domain size \\
`𝒰` = diffusive speed

`'advective_diffusive'` \\
`ℒ` = diffusive length \\
`𝒰` = advective speed
"""


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
    D_adv: FiniteDifference | FiniteDifferenceArgwise = FE,
    D_diff: FiniteDifference = FE,
    # TODO supg stabilization
    # c_stabilization: str | None = None,
    # c_limits: tuple[float, float] | EllipsisType | None = None,
    # linear algebra
    psi_petsc: OptionsPETSc = OptionsPETSc('cg', 'gamg'),
    c_petsc: OptionsPETSc = OptionsPETSc('gmres', 'ilu'),
    # optional postprocessing
    diagnostic: bool | Iterable[Solver] = False,    
    namespace: Iterable = (),
) -> Simulation:
    """
    `ϕ∂c/∂t + 𝐮·∇c =  ∇·(D(ϕ,𝐮)·∇c) ` \\
    `∇⋅𝐮 = 0` \\
    `𝐮 = -(K(ϕ)/μ(c))·(∇p - ρ(c)e₉)` 
    """
    # time
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(Omega, "t", order, ics=0.0)  
    dt = ConstantSeries(Omega, 'dt')
    # default boundary conditions
    if c_bcs is Ellipsis:
        c_bcs = BoundaryConditions(("neumann", dOmega.union, 0.0))
    if psi_bcs is Ellipsis:
        psi_bcs = BoundaryConditions(("dirichlet", dOmega.union, 0.0))
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
    if diagnostic:
        uMinMax = ConstantSeries(Omega, "uMinMax", shape=(2,))
        uRMS = ConstantSeries(Omega, 'uRMS')
        rms_norm = 2
        cMinMax = ConstantSeries(Omega, "cMinMax", shape=(2,))
        dtCFL = ConstantSeries(Omega, "dtCFL")
        fBoundary = ConstantSeries(Omega, "fBoundary", shape=(len(dOmega.markers), 2))
        solvers.extend(
            [
                evaluation(uMinMax, extrema)(u[0]),
                integration(uRMS, L_norm, 'dx', norm=rms_norm)(sqrt(inner(u[0], u[0])), rms_norm),
                evaluation(cMinMax, extrema)(c[0]),
                evaluation(dtCFL, cfl_timestep)(u[0], cfl_h),
                integration(fBoundary, flux, 'ds', *dOmega.markers)(c[0], u[0], FE(d)),
            ]
        )
        if isinstance(diagnostic, Iterable):
            solvers.extend(diagnostic)
            
    namespace = [phi, ('k', k), ('d', d), rho, mu, *namespace]
    return Simulation(solvers, t, dt, namespace)