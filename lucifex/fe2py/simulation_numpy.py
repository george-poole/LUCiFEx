from abc import abstractmethod
from typing import Generic, TypeVar
from collections.abc import Iterable

import numpy as np
from ufl.core.expr import Expr

from ..fem import Constant
from .series import NumpySeries, TriSeries, GridSeries, FloatSeries, as_numpy_series
from ..utils import is_cartesian, is_simplicial, create_fem_function, grid
from ..utils.py_utils import MultiKey

from ..sim.simulation import Simulation
        

T = TypeVar('T') # FIXME bounded by NumpySeriesABC
class NumpySimulationABC(
    Generic[T],
    MultiKey[str, T]
):
    def __init__(
        self,
        series: Iterable[T | FloatSeries],
        auxiliary: Iterable[T | FloatSeries | dict[str, np.ndarray | float]] = (),
        timings = None,
    ):
        self._series = series
        self._auxiliary = auxiliary
        self._timings = timings

    def _getitem(
        self, 
        key,
    ):
        return self.namespace[key]

    @property
    def series(self) -> list[T| FloatSeries]:
        return list(self._series)

    @property
    def namespace(self) -> dict[str, T | FloatSeries | float]:
        return {i.name: i for i in self._series} # TODO expr and consts too
    
    @classmethod
    def from_simulation(
        cls,
        sim: Simulation,
        slc: slice = slice(None, None, None),
        auxiliary: bool = False, #FIXME
        use_cache: tuple[bool, bool] = (True, True),
    ):
        _series = []
        for s in sim.solutions:
            s_np = as_numpy_series(s, use_cache=use_cache, slc=slc)
            _series.append(s_np)

        _auxiliary = []
        if auxiliary:
            for aux in sim.auxiliary:
                if isinstance(aux, Expr):
                    aux_func = create_fem_function(...)
                    aux_np = grid(...)(aux_func)
                    _auxiliary.append(aux_np)
                if isinstance(aux, Constant):
                    _auxiliary.append

        return cls(_series, _auxiliary, sim.timings)


class GridSimulation(NumpySimulationABC[GridSeries]):
    ...


class TriangulationSimulation(NumpySimulationABC[TriSeries]):
    ...



def numpy_simulation(
    sim: Simulation,
) -> GridSimulation | TriangulationSimulation:
    
    assert sim.mesh is not None
    simplicial = is_simplicial(use_cache=True)(sim.mesh)
    cartesian = is_cartesian(use_cache=True)(sim.mesh)
    
    match simplicial, cartesian:
        case True, False:
            ...


    