from .bcs import BoundaryConditions, BoundaryType, create_marked_measure
from .pde import (
    BoundaryValueProblem,
    InitialBoundaryValueProblem,
    InitialValueProblem,
    bvp_solver,
    ibvp_solver,
    ivp_solver,
)
from .options import OptionsJIT, OptionsFFCX, OptionsPETSc
from .evaluation import (
    EvaluationProblem,
    FacetIntegrationProblem,
    CellIntegrationProblem,
    InteriorFacetIntegrationProblem,
    dx_solver,
    ds_solver,
    dS_solver,
    eval_solver,
    InterpolationProblem,
    interpolation_solver,
    ProjectionProblem,
    projection_solver,
)


BVP = BoundaryValueProblem
"""Alias to `BoundaryValueProblem`"""

IBVP = InitialBoundaryValueProblem
"""Alias to `InitialBoundaryValueProblem`"""

IVP = InitialValueProblem
"""Alias to `InitialValueProblem`"""

XP = EvaluationProblem
"""Alias to `AuxiliaryProblem`"""

DX = CellIntegrationProblem
"""Alias to `DxSolver`"""

DS = FacetIntegrationProblem
"""Alias to `DsSolver`"""

PRJC = ProjectionProblem
"""Alias to `ProjectionSolver`"""

ITPL = InterpolationProblem
"""Alias to `InterpolationSolver`"""

BCS = BoundaryConditions
"""Alias to `BoundaryConditions`"""

from typing import TypeAlias

PDE: TypeAlias = BVP | IBVP | IVP
"""Alias to `BVP | IBVP | IVP`"""

Solver: TypeAlias = PDE | EvaluationProblem
"""Alias to `PDE | ExpressionSolver`"""