from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self

import numpy as np
from dolfinx.mesh import Mesh

from ..fdm import Series, SubSeriesError
from ..utils.py_utils import MultiKey
from ..utils.fenicsx_utils import get_component_functions


class FE2PyMesh(ABC):
    ...


class FE2PyFunction(ABC):
    ...


D = TypeVar('D')
T = TypeVar('T')
class FE2PySeries(ABC, Generic[D, T]):
    """
    Abstract base class for a Numpy-compatible series representing a time-dependent quantity.
    """
    def __init__(
        self, 
        series: Iterable[T], 
        t: Iterable[float],
        name: str | tuple[str, Iterable[str]] | None,
    ): 
        assert len(series) == len(t)
        self._series = list(series)
        self._time_series = list(t)

        if isinstance(name, tuple):
            name, subnames = name
            subnames = tuple(subnames)
        else:
            subnames = None

        if name is None:
            name = self.__class__.__name__

        self.name = name
        self._subnames = subnames
        self._create_subname = lambda i: self._subnames[i] if self._subnames else f'{self.name}_{i}'

    @classmethod
    @abstractmethod
    def from_series(
        cls, 
        cached_convert: Callable[..., Callable[[Series], T]] | Callable[..., Callable[[Mesh], D]],
        u: Series,
        use_cache: tuple[bool, bool] = (True, True),
        slc: slice = slice(None, None, None),
        **convert_kwargs,
    ) -> tuple[list[T], list[float], D, ]:
        use_mesh_cache, use_func_cache = use_cache

        match u.shape:
            case (_, ):
                series = [
                    np.array(
                        [
                            cached_convert(use_cache=use_func_cache)(j, **convert_kwargs) 
                            for j in get_component_functions(('P', 1), i, use_cache=Ellipsis)
                        ]
                    ) 
                    for i in u.series[slc]
                ]
            case ():
                series = [cached_convert(use_cache=use_func_cache)(i, **convert_kwargs) for i in u.series[slc]]
            case _:
                raise ScalarVectorError(u)
            
        return (
            series,
            u.time_series[slc],
            cached_convert(use_cache=use_mesh_cache)(u.mesh), 
        )
    
    @property
    def series(self) -> list[T]:
        return self._series

    @property
    def time_series(self) -> list[float]:
        return self._time_series
    
    @property
    def name(self) -> str | None:
        return self._name
    
    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        assert '__' not in value
        self._name = value
    
    @cached_property
    def shape(self) -> tuple[int, ...] | list[tuple[int, ...]] | None:
        if not self.series:
            return None
        _shape = lambda x: () if isinstance(x, (float, int)) else x.shape
        shapes = [_shape(i) for i in self.series]
        if len(set(shapes)) == 1:
            return shapes[0]
        else:
            return shapes
        
    @property
    def is_homogeneous(self) -> bool:
        return not isinstance(self.shape, list)
    
    @abstractmethod
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        ...
    
    def split(
        self,
        names: Iterable[str] | None = None,
    ) -> tuple[Self, ...]:
        if self.shape == ():
            raise SubSeriesError
        subseries_indices = tuple(range(self.shape[0]))
        if names is None:
            names = [f'{self.name}_{i}' for i in subseries_indices]
        return tuple(self.sub(i, n) for i, n in zip(subseries_indices, names, strict=True))
    

# FIXME
T = TypeVar('T') 
class FE2PySimulation(
    Generic[T],
    MultiKey[str, T]
):
    ...
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



