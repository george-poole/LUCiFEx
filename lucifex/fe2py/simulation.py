from abc import abstractmethod
from typing import Generic, TypeVar
from collections.abc import Iterable

import numpy as np
from ufl.core.expr import Expr

from ..utils.fenicsx_utils import is_cartesian, is_simplicial, create_function
from ..utils.py_utils import MultiKey
from ..fem import Constant
from .tri import TriSeries
from .grid import GridSeries, as_grid_function
from .array import FloatSeries
from .generic import as_numpy_series
from .abc import FE2PySimulation


from ..sim.simulation import Simulation
        

class GridSimulation(FE2PySimulation[GridSeries]):
    ...


class TriSimulation(FE2PySimulation[TriSeries]):
    ...



def numpy_simulation(
    sim: Simulation,
) -> GridSimulation | TriSimulation:
    
    assert sim.mesh is not None
    simplicial = is_simplicial(use_cache=True)(sim.mesh)
    cartesian = is_cartesian(use_cache=True)(sim.mesh)
    
    match simplicial, cartesian:
        case True, False:
            ...


    