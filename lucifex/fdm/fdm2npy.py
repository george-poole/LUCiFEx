import operator
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable
from typing_extensions import Self

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh

from ..fem import Function, Constant
from ..fem.fem2npy import NPyMesh, NPyFunction, GridMesh, GridFunction, as_grid_function, TriMesh, TriFunction, as_tri_function
from . import Series, ExprSeries, ConstantSeries
from ..utils.py_utils import replicate_callable
from ..utils.fenicsx_utils import NonScalarVectorError
from ..utils.fenicsx_utils import get_component_functions
from .series import FunctionSeries


M = TypeVar('M', bound=NPyMesh)
F = TypeVar('F', bound=NPyFunction)
class NPySeries(ABC, Generic[M, F]):
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
    
    def sub(
        self: 'NPySeries', 
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
    NPySeries[GridMesh, GridFunction]
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
    

class TriSeries(
    NPySeries[TriMesh, TriFunction]
):            
    @classmethod
    def from_series(
        cls: type['GridSeries'], 
        u: FunctionSeries,
        slc: slice = slice(None, None, None),
        use_mesh_cache: bool = True,
        use_func_cache: bool = True,
    ) -> Self:
        convert_func = lambda u: (
            as_tri_function(use_cache=use_func_cache)(
                u, use_mesh_cache,
            )
        )
        return super().from_series(
            u, 
            convert_func,
            slc,
        )


class FloatSeries(NPySeries[None, int | float | np.ndarray]):
    @classmethod
    def from_series(
        cls, 
        u: ConstantSeries,
        slc: slice = slice(None, None, None),
    ) -> Self:
        return cls(u.value_series[slc], u.time_series[slc], u.name)
    
    
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
        

@replicate_callable(GridSeries.from_series)
def as_grid_series():
    pass


@replicate_callable(TriSeries.from_series)
def as_tri_series():
    pass


@replicate_callable(FloatSeries.from_series)
def as_float_series():
    pass



# @overload
# def as_numpy_series(
#     u: ConstantSeries,
#     slc: slice = slice(None, None, None),
#     use_cache: tuple[bool, bool] = (True, True),
# ) -> FloatSeries:
#     ...


# @overload
# def as_numpy_series(
#     u: FunctionSeries,
#     slc: slice = slice(None, None, None),
#     use_cache: tuple[bool, bool] = (True, True),
# ) -> GridSeries | TriSeries:
#     ...


# def as_numpy_series(
#     u: FunctionSeries| ConstantSeries,
#     *,
#     slc: slice = slice(None, None, None),
#     use_cache: tuple[bool, bool] = (True, True),
# ) :
#     if isinstance(u, ConstantSeries):
#         return FloatSeries.from_series(u, slc)
#     else:
#         simplicial = is_simplicial(use_cache=True)(u.mesh)
#         cartesian = is_grid(use_cache=True)(u.mesh)

#         match simplicial, cartesian:
#             case True, False:
#                 return TriSeries.from_series(u, use_cache, slc)
#             case _, True:
#                 return GridSeries.from_series(u, use_cache, slc)
#             case False, False:
#                 raise NonCartesianQuadMeshError