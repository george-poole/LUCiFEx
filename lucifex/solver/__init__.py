from .bcs import BoundaryConditions, BoundaryType, create_enumerated_measure
from .pde import (
    BoundaryValueProblem,
    InitialBoundaryValueProblem,
    InitialValueProblem,
    EigenvalueProblem,
    bvp_solver,
    ibvp_solver,
    ivp_solver,
    evp_solver,
)
from .options import OptionsJIT, OptionsFFCX, OptionsPETSc, OptionsSLEPc
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

EVP = EigenvalueProblem
"""Alias to `EigenvalueProblem`"""

from typing import TypeAlias

PDE: TypeAlias = BVP | IBVP | IVP | EVP
"""Alias to `BVP | IBVP | IVP`"""

Problem: TypeAlias = PDE | EvaluationProblem
"""Alias to `PDE | ExpressionSolver`"""