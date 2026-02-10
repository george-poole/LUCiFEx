from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..fdm.series_numpy import TriangulationSeries, GridSeries, numpy_series, NumericSeries
from ..utils import is_cartesian, is_simplicial

from .simulation import Simulation
        

T = TypeVar('T')
class NumpySimulationABC(ABC, Generic[T]):

    def __init__(
        self,
        series: dict[str, T | NumericSeries],
        exprs_consts = (),
        timings = None,
    ):
        self._series = series
        self._exprs_consts = exprs_consts
        self._timings = timings

    def __getitem__(
        self, 
        key: str | tuple[str, ...],
    ):
        ...

    @property
    def series(self) -> list[T| NumericSeries]:
        ...

    @property
    def namespace(self) -> dict[str, T | NumericSeries | float]:
        return self._series # TODO expr and consts too
    

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
            s_np = numpy_series(s, use_cache=use_cache, slc=slc)
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


    