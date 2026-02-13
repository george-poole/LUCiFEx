from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh

from ..fem import Function, Constant
from ..fdm import Series, ExprSeries, ConstantSeries
from ..sim import Simulation
from ..utils.py_utils import MultiKey
from ..utils.fenicsx_utils import NonScalarVectorError
from ..utils.fenicsx_utils import get_component_functions
from .array import FloatSeries


M = TypeVar('M', bound=FE2PyMesh)
F = TypeVar('F', bound=FE2PyFunction)
S = TypeVar('S', bound=FE2PySeries)
class FE2PySimulation(
    Generic[S],
    MultiKey[str, S | F | FloatSeries | np.ndarray | float]
):
    def __init__(
        self,
        solutions: Iterable[S | FloatSeries],
        auxiliary: Iterable[S | F | FloatSeries | tuple[str, np.ndarray | float]] = (),
        timings: dict | None = None,
    ):
        self._solutions = list(solutions)
        self._auxiliary = list(auxiliary)
        self._timings = timings

    def _getitem(
        self, 
        key,
    ):
        return self.namespace[key]

    @property
    def solutions(self) -> list[S| FloatSeries]:
        return list(self._solutions)
    
    @property
    def auxiliary(self) -> list[S| FloatSeries | tuple[str, np.ndarray | float]]:
        return self._auxiliary

    @property
    def namespace(self) -> dict[str, F | FloatSeries | float]:
        d = {i.name: i for i in self._solutions}
        d.update({f.name for f in self._auxiliary if not isinstance(f, tuple)})        
        d.update({f[0]: f[1] for f in self._auxiliary if isinstance(f, tuple)})
        return d

    @property
    def timings(self) -> dict | None:
        return self._timings
    
    @property
    def mesh(self) -> M:
        return ...
    
    @classmethod
    @abstractmethod
    def from_simulation(
        cls,
        sim: Simulation,
        convert_func: Callable[[Series], F],
        auxiliary: bool = False, 
    ):
        _solutions = []
        for s in sim.solutions:
            if isinstance(s, ConstantSeries):
                _sltn = ...
            else:
                _sltn = convert_func(s)
            _solutions.append(_sltn)

        _auxiliary = []
        if auxiliary:
            for aux in sim.auxiliary:
                if isinstance(aux, ExprSeries):
                    raise NotImplementedError
                elif isinstance(aux, Expr):
                    raise NotImplementedError
                elif isinstance(aux, Function):
                    raise NotImplementedError
                elif isinstance(aux, Constant):
                    _aux = (aux.name, aux.value)
                elif isinstance(aux, tuple):
                    _aux = aux
                else:
                    raise TypeError
                _auxiliary.append(_aux)

        return cls(_solutions, _auxiliary, sim.timings)
    

# # FIXME
# T = TypeVar('T') 
# class FE2PySimulation(
#     Generic[T],
#     MultiKey[str, T]
# ):
#     ...
    # def __init__(
    #     self,
    #     series: Iterable[T | FloatSeries],
    #     auxiliary: Iterable[T | FloatSeries | dict[str, np.ndarray | float]] = (),
    #     timings = None,
    # ):
    #     self._series = series
    #     self._auxiliary = auxiliary
    #     self._timings = timings

    # def _getitem(
    #     self, 
    #     key,
    # ):
    #     return self.namespace[key]

    # @property
    # def series(self) -> list[T| FloatSeries]:
    #     return list(self._series)

    # @property
    # def namespace(self) -> dict[str, T | FloatSeries | float]:
    #     return {i.name: i for i in self._series} # TODO expr and consts too
    
    # @classmethod
    # def from_simulation(
    #     cls,
    #     sim: Simulation,
    #     slc: slice = slice(None, None, None),
    #     auxiliary: bool = False, #FIXME
    #     use_cache: tuple[bool, bool] = (True, True),
    # ):
    #     _series = []
    #     for s in sim.solutions:
    #         s_np = as_numpy_series(s, use_cache=use_cache, slc=slc)
    #         _series.append(s_np)

    #     _auxiliary = []
    #     if auxiliary:
    #         for aux in sim.auxiliary:
    #             if isinstance(aux, Expr):
    #                 aux_func = create_fem_function(...)
    #                 aux_np = grid(...)(aux_func)
    #                 _auxiliary.append(aux_np)
    #             if isinstance(aux, Constant):
    #                 _auxiliary.append

    #     return cls(_series, _auxiliary, sim.timings)



