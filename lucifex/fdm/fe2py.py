from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh

from ..fem import Function, Constant
from ..fem.fem2py import FE2PyMesh, FE2PyFunction, GridMesh, GridFunction
from . import Series, ExprSeries, ConstantSeries
from ..utils.py_utils import replicate_callable
from ..utils.fenicsx_utils import NonScalarVectorError
from ..utils.fenicsx_utils import get_component_functions
from .series import FunctionSeries


M = TypeVar('M', bound=FE2PyMesh)
F = TypeVar('F', bound=FE2PyFunction)
class FE2PySeries(ABC, Generic[M, F]):
    """
    Abstract base class for a Numpy-compatible series representing a time-dependent quantity.
    """
    def __init__(
        self, 
        series: Iterable[F] | Iterable[Iterable[F]], 
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
        self._create_subname = lambda i: (
            self._subnames[i] if self._subnames else f'{self.name}{i}'
        )

    @classmethod
    @abstractmethod
    def from_series(
        cls, 
        u: Series,
        convert_func: Callable[[Function], F],
        slc: slice = slice(None, None, None),
    ) -> Self:
        match u.shape:
            case (_, ):
                series = [
                    [convert_func(j) for j in get_component_functions(('P', 1), i, use_cache=Ellipsis)] 
                    for i in u.series[slc]
                ]
            case ():
                series = [convert_func(i) for i in u.series[slc]]
            case _:
                raise NonScalarVectorError(u)
            
        return cls(
            series,
            u.time_series[slc],
            u.name,
        )
    
    @abstractmethod
    def sub(
        self: 'FE2PySeries', 
        index: int, 
        name: str | None = None,
    ) -> Self:
        assert not self.is_scalar
        if name is None:
            name = self._create_subname(index)
        return self.__class__(
            [i[index] for i in self.series], 
            self.time_series, 
            name,
        )
    
    def split(
        self,
        names: Iterable[str] | None = None,
    ) -> tuple[Self, ...]:
        assert not self.is_scalar
        n_sub = len(self.series[0])
        subseries_indices = range(len(self.series[0]))
        if names is None:
            names = [f'{self.name}_{i}' for i in subseries_indices]
        return tuple(self.sub(i, n) for i, n in zip(range(n_sub), names, strict=True))
    
    @property
    def series(self) -> list[F] | list[list[F]]:
        return self._series

    @property
    def time_series(self) -> list[float]:
        return self._time_series
    
    @property
    def mesh(self) -> M:
        return ...
    
    @property
    def name(self) -> str | None:
        return self._name
    
    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        assert '__' not in value
        self._name = value

    @cached_property
    def is_scalar(self):
        return all(isinstance(i, F.__bound__) for i in self.series)
    
    # @cached_property
    # def shape(self) -> tuple[int, ...] | list[tuple[int, ...]]:
    #     shapes = [] 
    #     for i in self.series:
    #         if isinstance(i, FE2PyFunction):
    #             shp = i.shape
    #         else:
    #             shp = [j.shape for j in i]
    #         shapes.append(shp)

    #     if len(set(shapes)) == 1:
    #         return shapes[0]
    #     else:
    #         return shapes
        
    # @property
    # def is_homogeneous(self) -> bool: #FIXME
    #     return not isinstance(self.shape, list)
    

class GridSeries(
    FE2PySeries[GridMesh, GridFunction]
):            
    @classmethod
    def from_series(
        cls: type['GridSeries'], 
        u: FunctionSeries,
        slc: slice = slice(None, None, None),
        strict: bool = False,
        jit: bool = True,
        mask: float = np.nan,
        use_mesh_map: bool = False,
        use_mesh_cache: bool = True,
        use_func_cache: bool = True,
    ) -> Self:
        convert_func = lambda u: (
            as_grid_function(use_cache=use_func_cache)(
                u, strict, jit, mask, use_mesh_map, use_mesh_cache
            )
        )
        return super().from_series(
            u, 
            convert_func,
            slc,
        )
    

@replicate_callable(GridSeries.from_series)
def as_grid_series():
    pass


