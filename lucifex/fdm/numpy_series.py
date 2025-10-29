from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic
from typing_extensions import Self

import numpy as np
from matplotlib.tri.triangulation import Triangulation

from ..utils import grid, triangulation
from ..utils.fem_utils import ScalarVectorError
from ..utils.fem_typecast import finite_element_function_components
from .series import ConstantSeries, FunctionSeries, SubSeriesError


T = TypeVar('T')
class NumpySeriesABC(ABC, Generic[T]):
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
        *args, 
        **kwargs,
    ) -> Self:
        ...
    
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


class NumericSeries(NumpySeriesABC[int | float | np.ndarray]):
    @classmethod
    def from_series(
        cls, 
        u: ConstantSeries,
    ) -> Self:
        return cls(u.value_series, u.time_series, u.name)
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        if self.shape == ():
            raise SubSeriesError
        if name is None:
            name = self._create_subname(index)
        return NumericSeries([i[index] for i in self.series], self.time_series, name)
    

class GridSeries(NumpySeriesABC[np.ndarray]):
    def __init__(
        self, 
        series: list[np.ndarray], 
        t: list[float], 
        x: tuple[np.ndarray, ...],
        name: str | None = None,
    ): 
        super().__init__(series, t, name)
        self._axes = x
    
    @classmethod
    def from_series(
        cls, 
        u: FunctionSeries,
        use_cache: tuple[bool, bool] = (True, True),
        **grid_kwargs,
    ) -> Self:
        use_mesh_cache, use_func_cache = use_cache

        match u.shape:
            case ():
                series = [grid(use_cache=use_func_cache)(i, **grid_kwargs) for i in u.series]
            case (_, ):

                series = [
                    np.array(
                        [
                            grid(use_cache=use_func_cache)(j, **grid_kwargs) 
                            for j in finite_element_function_components(('P', 1), i, use_cache=Ellipsis)
                        ]
                    ) 
                    for i in u.series
                ]
            case _:
                raise ScalarVectorError(u)
            
        return cls(
            series,
            u.time_series,
            grid(use_cache=use_mesh_cache)(u.mesh), 
            u.name,
        )
    
    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        return self._axes
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        if name is None:
            name = self._create_subname(index)
        return GridSeries([i[index] for i in self.series], self.time_series, self.axes, name)
    

class TriangulationSeries(NumpySeriesABC[np.ndarray]):
    def __init__(
        self, 
        series: list[np.ndarray], 
        t: list[float], 
        triangulation: Triangulation,
        name: str | None = None,
    ): 
        super().__init__(series, t, name)
        self._triangulation = triangulation

    @property
    def triangulation(self) -> Triangulation:
        return self._triangulation
    
    @classmethod
    def from_series(
        cls, 
        u: FunctionSeries,
        use_cache: tuple[bool, bool] = (True, True),
    ) -> Self:
        use_mesh_cache, use_func_cache = use_cache

        match u.shape:
            case ():
                series = [triangulation(use_func_cache=use_func_cache)(i) for i in u.series]
            case (_, ):
                series = [
                    np.array(
                        [
                            triangulation(use_func_cache=use_func_cache)(j) 
                            for j in finite_element_function_components(('P', 1), i, use_cache=(True, True))
                        ]
                    )
                    for i in u.series
                ]
            case _:
                raise ScalarVectorError(u)
            
        return cls(
            series,
            u.time_series,
            triangulation(use_cache=use_mesh_cache)(u.mesh), 
            u.name,
        )
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        if name is None:
            name = self._create_subname(index)
        return TriangulationSeries([i[index] for i in self.series], self.time_series, self.triangulation, name)