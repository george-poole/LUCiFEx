from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self
import operator

from dolfinx.fem import Function

from ..fdm import ConstantSeries, FunctionSeries, Series, SubSeriesError
from ..utils.fenicsx_utils import is_cartesian, is_simplicial, NonCartesianQuadMeshError
from .grid import GridFunction, as_grid_function, GridSeries
from .tri import TriFunction, as_tri_function, TriSeries
from .array import FloatSeries


def as_numpy_function(
    u: Function,
    cartesian: bool | None = None,
    use_cache: bool | tuple[bool, bool] = (True, False),
) -> GridFunction | TriFunction:
    
    mesh = u.function_space.mesh

    if isinstance(use_cache, bool):
        use_cache = (use_cache, use_cache)
    use_mesh_cache, use_func_cache = use_cache

    if cartesian is None:
        cartesian = is_cartesian(use_cache=use_mesh_cache)(mesh)
    simplicial = is_simplicial(use_cache=use_mesh_cache)(mesh)

    match simplicial, cartesian:
        case True, False:
            return as_tri_function(use_cache=use_func_cache)(u)
        case _, True:
            return as_grid_function(use_cache=use_func_cache)(u)
        case False, False:
            raise NonCartesianQuadMeshError
        


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
) -> GridSeries | TriSeries:
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
        cartesian = is_cartesian(use_cache=True)(u.mesh)

        match simplicial, cartesian:
            case True, False:
                return TriSeries.from_series(u, use_cache, slc)
            case _, True:
                return GridSeries.from_series(u, use_cache, slc)
            case False, False:
                raise NonCartesianQuadMeshError