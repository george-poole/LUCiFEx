from .bcs import BoundaryConditions, BoundaryType, create_tagged_measure
from .pde import (
    BoundaryValueProblem,
    InitialBoundaryValueProblem,
    InitialValueProblem,
    EigenvalueProblem,
    Projection,
    bvp,
    ibvp,
    ivp,
    evp,
    projection,
)
from .options import OptionsJIT, OptionsFFCX, OptionsPETSc, OptionsSLEPc
from .eval import (
    Evaluation,
    Interpolation,
    Integration,
    evaluation,
    interpolation,
    integration,
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

Solver: TypeAlias = PDE | Evaluation
"""Alias to `PDE | EvaluationProblem`"""