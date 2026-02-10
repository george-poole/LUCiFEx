from abc import abstractmethod
from typing import Generic, TypeVar
from collections.abc import Iterable

from ..fdm.series_numpy import NumpySeriesABC, TriangulationSeries, GridSeries, FloatSeries, as_numpy_series
from ..utils import is_cartesian, is_simplicial
from ..utils.py_utils import MultiKey

from .simulation import Simulation
        

T = TypeVar('T')
class NumpySimulationABC(
    Generic[T],
    MultiKey[str, T]
):

    def __init__(
        self,
        series: Iterable[T | FloatSeries],
        exprs_consts = (),
        timings = None,
    ):
        self._series = series
        self._exprs_consts = exprs_consts
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
    

class GridSimulation(NumpySimulationABC[GridSeries]):

    @classmethod
    def from_simulation(
        cls,
        sim: Simulation,
        use_cache: tuple[bool, bool] = (True, True),
        slc: slice = slice(None, None, None)
    ):
        series = {}
        for s in sim.series:
            s_np = as_numpy_series(s, use_cache=use_cache, slc=slc)
            series[s.name] = s_np

        # exprs_consts = sim.exprs_consts
        # timings = sim.timings

        return cls(series)


class TriangulationSimulation(NumpySimulationABC[TriangulationSeries]):

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


    