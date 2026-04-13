from typing import Callable

import numpy as np

from lucifex.mesh import rectangle_mesh, mesh_boundary
from lucifex.fem import  Constant, SpatialPerturbation, cubic_noise
from lucifex.fdm import (
    ConstantSeries, FunctionSeries, finite_difference_order,
    FiniteDifference, FiniteDifferenceArgwise, AB2, CN,
)
from lucifex.solver import BoundaryConditions, OptionsPETSc, evaluation
from lucifex.sim import configure_simulation
from lucifex.utils.fenicsx_utils import CellType

from .C10_darcy_convection_generic import darcy_convection_generic, DARCY_CONVECTION_SCALINGS


@configure_simulation(
    store_delta=1,
    write_delta=None,
)
def darcy_convection_data_driven(
    aspect: float = 1.0,
    Nx: int = 100,
    Ny: int = 100,
    cell: str = CellType.QUADRILATERAL,
    scaling: str = 'advective',
    Ra: float = 5e2,
    porosity: Callable[[np.ndarray], np.ndarray] | float = 1,
    permeability: Callable = lambda phi: phi**2,
    c_lower: Callable[
       [float | Constant], 
       Callable[[np.ndarray], np.ndarray]
    ] = lambda t: float(t),
    c_eps: float = 1e-2,
    c_ampl: float = 1e-6,
    c_freq: tuple[int, int] = (8, 8),
    c_seed: tuple[int, int] = (123, 321),
    dt_max: float = 0.5,
    dt_min: float = 0.0,
    dt_h: str | float = "hmin",
    dt_Cu: float = 0.75,
    D_adv: FiniteDifference | FiniteDifferenceArgwise = (AB2 @ CN),
    D_diff: FiniteDifference = CN,
    psi_petsc: OptionsPETSc | None = None,
    c_limits: tuple[float, float] | bool = False,
    c_petsc: OptionsPETSc | None = None,
    diagnostic: bool = False,
):
    # space
    scaling_map = DARCY_CONVECTION_SCALINGS[scaling](Ra)
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
    # constants
    Di, Bu = scaling_map(Omega)['Di', 'Bu']
    Ra = Constant(Omega, Ra, 'Ra')
    # boundary conditions
    order = finite_difference_order(D_adv, D_diff)
    t = ConstantSeries(Omega, "t", order, ics=0.0)
    cD = FunctionSeries(
        (Omega, 'P', 1), 
        'cD', 
        order=order,
        ics=c_lower(0.0),
    )
    cD_solver = evaluation(cD, c_lower, future=True)(t[1])
    c_bcs = BoundaryConditions(
        ("dirichlet", dOmega['lower'], cD[0]),
        ("dirichlet", dOmega['upper'], 0.0),
        # ('neumann', dOmega['left', 'right'], 0.0)
    )
    # initial conditions
    c_ics = SpatialPerturbation(
        lambda x: c_lower(0.0)(x) * np.exp(-x[1] / c_eps),
        cubic_noise(['neumann', 'dirichlet'], [Lx, Ly], c_freq, c_seed),
        [Lx, Ly],
        c_ampl,
    ) 
    # constitutive relations
    dispersion = lambda phi: Di * phi
    density = lambda c: -Bu * c
    return darcy_convection_generic(
        Omega=Omega, 
        dOmega=dOmega, 
        t=t,
        c_ics=c_ics, 
        c_bcs=c_bcs, 
        porosity=porosity,
        permeability=permeability,
        dispersion=dispersion,
        density=density, 
        dt_max=dt_max, 
        dt_min=dt_min,
        dt_h=dt_h, 
        dt_Cu=dt_Cu,
        D_adv=D_adv, 
        D_diff=D_diff, 
        psi_petsc=psi_petsc, 
        c_petsc=c_petsc,
        c_limits=c_limits, 
        diagnostic=diagnostic,
        post_solvers=[cD_solver],
        c_fs=cD.function_space,
    )