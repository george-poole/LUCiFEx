from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable
from typing_extensions import Self

import numpy as np

from ..mesh.mesh2npy import NPyMesh, GridMesh, TriMesh
from ..fem import Function, Constant
from ..fem.fem2npy import (
    NPyNamedObject, NPyFunction, GridFunction, as_grid_function, 
    TriFunction, as_tri_function, NPyConstant, as_npy_constant,
)
from . import Series, ExprSeries, ConstantSeries
from ..utils.py_utils import replicate_callable
from .series import FunctionSeries


F = TypeVar('F', bound=NPyNamedObject)
class NPySeries(NPyNamedObject, Generic[F]):
    def __init__(
        self, 
        series: Iterable[F], 
        t: Iterable[float],
        name: str | tuple[str, Iterable[str]] | None,
    ): 
        super().__init__(name)
        assert len(series) == len(t)
        self._series = list(series)
        self._time_series = list(t)

    @property
    def series(self) -> list[F]:
        return self._series

    @property
    def time_series(self) -> list[float]:
        return self._time_series
    
    @property
    def ufl_shape(self) -> tuple[int, ...] | None:
        if self.series:
            return self.series[0].ufl_shape
        
    # @property
    # def npy_shape(self) -> tuple[int, ...] | None:
    #     if self.series:
    #         return self.series[0].npy_shape

    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self | None:
        if self.ufl_shape == ():
            return None
        subseries = [i.sub(index) for i in self.series]
        return self.__class__(
            subseries,
            self.time_series,
            self._create_subname(index) if name is None else name,
        )

    @classmethod
    @abstractmethod
    def from_series(
        cls,
        u: Series,
        *args,
        **kwargs,
    ):
        ...


class NPyConstantSeries(
    NPySeries[NPyConstant]
):
    @classmethod
    def from_series(
        cls, 
        c: ConstantSeries,
        slc: slice = slice(None, None, None),
    ) -> Self:
        series = [as_npy_constant(ci) for ci in c.series[slc]]
        return cls(series, c.time_series, c.name)
    


M = TypeVar('M', bound=NPyMesh)
F = TypeVar('F', bound=NPyFunction)
class NPyFunctionSeries(NPySeries[F], Generic[M, F]):
    @classmethod
    @abstractmethod
    def from_series(
        cls, 
        u: Series,
        convert_func: Callable[[Function], F],
        slc: slice = slice(None, None, None),
    ) -> Self:
        series = [convert_func(ui) for ui in u.series[slc]]
        return cls(
            series,
            u.time_series[slc],
            u.name,
        )
    
    @property
    def mesh(self) -> M | None:
        if self.series:
            return self.series[0].mesh
    

class GridFunctionSeries(
    NPyFunctionSeries[GridMesh, GridFunction]
):            
    @classmethod
    def from_series(
        cls: type['GridFunctionSeries'], 
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
    

class TriFunctionSeries(
    NPyFunctionSeries[TriMesh, TriFunction]
):            
    @classmethod
    def from_series(
        cls: type['GridFunctionSeries'], 
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


    # def transform(
    #     self, 
    #     transf: str | Callable[[float | Iterable[float]], float] | Callable[[float, float], float],
    #     other: Self | float | None = None,
    #     name: str | None = None,
    # ) -> Self:
    #     if isinstance(transf, str):
    #         transf = getattr(operator, transf)

    #     if other is None:
    #         return NPyConstantSeries([transf(i) for i in self.series], self.time_series, name)
    #     elif isinstance(other, float):
    #         return NPyConstantSeries([transf(i, other) for i in self.series], self.time_series, name)
    #     else:
    #         size = min(len(self.time_series), len(other.time_series))
    #         assert np.allclose(self.time_series[:size], other.time_series[:size])
    #         return NPyConstantSeries(
    #             [transf(i, j) for i, j in zip(self.series[:size], other.series[:size])],
    #             self.time_series[:size],
    #             name,
    #         )
        
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
        

@replicate_callable(NPyConstantSeries.from_series)
def as_npy_constant_series():
    pass


@replicate_callable(GridFunctionSeries.from_series)
def as_grid_function_series():
    pass


@replicate_callable(TriFunctionSeries.from_series)
def as_tri_function_series():
    pass


def as_npy_function_series():
    ...


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