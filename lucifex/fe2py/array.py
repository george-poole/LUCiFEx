from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self
import operator

import numpy as np
from dolfinx.fem import Function

from ..fdm import ConstantSeries, FunctionSeries, Series, SubSeriesError
from ..utils.fenicsx_utils import is_cartesian, is_simplicial, NonCartesianQuadMeshError
from .grid import GridFunction, as_grid_function, GridSeries
from .tri import TriFunction, as_tri_function, TriSeries
from .abc import FE2PySeries


class FloatSeries(FE2PySeries[None, int | float | np.ndarray]):
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
        

