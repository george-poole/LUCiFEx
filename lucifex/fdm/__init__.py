"""
Finite differences in time
"""
from .finite_difference import (
    FiniteDifference, FiniteDifferenceDerivative,
    BDF, AB, AM, AB1, AB2, BE, FE, CN, DT, DT2, DTLF, AM1, AM2,
    BDF1, BDF2, finite_difference_order, finite_difference_discretize,
)
from .series import FunctionSeries, ConstantSeries, Series, ExprSeries, GridSeries, NumericSeries
from .ufl_operators import inner, grad, curl, div, testfunction, trialfunction
from .timestep import cfl_timestep, kinetic_timestep, cflk_timestep