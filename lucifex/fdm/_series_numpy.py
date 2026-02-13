from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self
import operator

import numpy as np
from matplotlib.tri.triangulation import Triangulation
from dolfinx.mesh import Mesh

from . import grid, tri
from ..utils.fenicsx_utils import NonScalarVectorError, get_component_functions, is_grid, is_simplicial, NonCartesianQuadMeshError
from .series import ConstantSeries, FunctionSeries, Series, SubSeriesError


D = TypeVar('D')
T = TypeVar('T')
class NumpySeries(ABC, Generic[D, T]):
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
                raise NonScalarVectorError(u)
            
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


class FloatSeries(NumpySeries[int | float | np.ndarray]):
    @classmethod
    def from_series(
        cls, 
        u: ConstantSeries,
        slc: slice = slice(None, None, None),
    ) -> Self:
        return cls(u.value_series[slc], u.time_series[slc], u.name)
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        if self.shape == ():
            raise SubSeriesError
        if name is None:
            name = self._create_subname(index)
        return FloatSeries([i[index] for i in self.series], self.time_series, name)
    
    def transform(
        self, 
        transf: str | Callable[[float | Iterable[float]], float] | Callable[[float, float], float],
        other: Self | float | None = None,
        name: str | None = None,
    ) -> Self:
        if isinstance(transf, str):
            transf = getattr(operator, transf)

        if other is None:
            return FloatSeries([transf(i) for i in self.series], self.time_series, name)
        elif isinstance(other, float):
            return FloatSeries([transf(i, other) for i in self.series], self.time_series, name)
        else:
            size = min(len(self.time_series), len(other.time_series))
            assert np.allclose(self.time_series[:size], other.time_series[:size])
            return FloatSeries(
                [transf(i, j) for i, j in zip(self.series[:size], other.series[:size])],
                self.time_series[:size],
                name,
            )
    

class GridSeries(
    NumpySeries[tuple[np.ndarray, ...], np.ndarray]
):
    def __init__(
        self, 
        series: list[np.ndarray], 
        t: list[float], 
        axes: tuple[np.ndarray, ...],
        name: str | None = None,
    ): 
        super().__init__(series, t, name)
        self._axes = axes
    
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
    
    @classmethod
    def from_series(
        cls: type['GridSeries'], 
        u: Series,
        use_cache: tuple[bool, bool] = (True, True),
        slc: slice = slice(None, None, None),
        **grid_kwargs,
    ):
        series, time_series, axes = super().from_series(
            grid,
            u,
            use_cache,
            slc,
            **grid_kwargs,
        )
        return cls(series, time_series, axes, u.name)
    

class TriangulationSeries(
    NumpySeries[Triangulation, np.ndarray]
):
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
        cls: type['TriangulationSeries'], 
        u: FunctionSeries,
        use_cache: tuple[bool, bool] = (True, True),
        slc: slice = slice(None, None, None),
        **tri_kwargs,
    ) -> Self:
        series, time_series, trigl = super().from_series(
            tri,
            u,
            use_cache,
            slc,
            **tri_kwargs,
        )
        return cls(series, time_series, trigl, u.name)
        
        
        
        # use_mesh_cache, use_func_cache = use_cache

        # match u.shape:
        #     case (_, ):
        #         series = [
        #             np.array(
        #                 [
        #                     triangulation(use_func_cache=use_func_cache)(j) 
        #                     for j in get_component_functions(('P', 1), i, use_cache=(True, True))
        #                 ]
        #             )
        #             for i in u.series[slc]
        #         ]
        #     case ():
        #         series = [triangulation(use_func_cache=use_func_cache)(i) for i in u.series[slc]]
        #     case _:
        #         raise ScalarVectorError(u)
            
        # return cls(
        #     series,
        #     u.time_series[slc],
        #     triangulation(use_cache=use_mesh_cache)(u.mesh), 
        #     u.name,
        # )
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,

    ) -> Self:
        if name is None:
            name = self._create_subname(index)
        return TriangulationSeries([i[index] for i in self.series], self.time_series, self.triangulation, name)
    

@overload
def as_numpy_series(
    u: ConstantSeries,
    slc: slice = slice(None, None, None),
    use_cache: tuple[bool, bool] = (True, True),
) -> FloatSeries:
    ...


@overload
def as_numpy_series(
    u: FunctionSeries,
    slc: slice = slice(None, None, None),
    use_cache: tuple[bool, bool] = (True, True),
) -> GridSeries | TriangulationSeries:
    ...


def as_numpy_series(
    u: FunctionSeries| ConstantSeries,
    *,
    slc: slice = slice(None, None, None),
    use_cache: tuple[bool, bool] = (True, True),
) :
    if isinstance(u, ConstantSeries):
        return FloatSeries.from_series(u, slc)
    else:
        simplicial = is_simplicial(use_cache=True)(u.mesh)
        cartesian = is_grid(use_cache=True)(u.mesh)

        match simplicial, cartesian:
            case True, False:
                return TriangulationSeries.from_series(u, use_cache, slc)
            case _, True:
                return GridSeries.from_series(u, use_cache, slc)
            case False, False:
                raise NonCartesianQuadMeshError

