from .constant import SpatialConstant
from .function import SpatialFunction
from .unsolved import Unsolved, UnsolvedType, is_unsolved

Function = SpatialFunction
"""
Alias to `lucifex.fem.function.SpatialFunction`, 
not to be confused with `dolfinx.fem.function.Function.`
"""

Constant = SpatialConstant
"""
Alias to `lucifex.fem.constant.SpatialConstant`, 
not to be confused with `dolfinx.fem.function.Constant.`
"""