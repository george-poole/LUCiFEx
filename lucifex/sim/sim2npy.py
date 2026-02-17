from abc import ABC, abstractmethod
from typing import TypeVar, Iterable, Generic, Callable
from typing_extensions import Self

from dolfinx.fem import Function
from ufl.core.expr import Expr
import numpy as np

from ..mesh.mesh2npy import NPyMesh, GridMesh, TriMesh
from ..fem import Function, Constant
from ..fem.fem2npy import NPyFunction, GridFunction, TriFunction, NPyConstant
from ..fdm import Series, FunctionSeries, ConstantSeries, ExprSeries
from ..fdm.fdm2npy import (
    NPyFunctionSeries, GridFunctionSeries, as_grid_function_series,
    as_tri_function_series, TriFunctionSeries, NPyConstantSeries, as_npy_constant_series,
)
from ..utils.py_utils import MultiKey
from ..utils.py_utils import replicate_callable
from .simulation import Simulation


M = TypeVar('M', bound=NPyMesh)
F = TypeVar('F', bound=NPyFunction)
S = TypeVar('S', bound=NPyFunctionSeries)
class NPySimulation(
    Generic[M, F, S],
    MultiKey[str, S | F | NPyConstantSeries | np.ndarray | float]
):
    def __init__(
        self,
        solutions: Iterable[S | NPyConstantSeries],
        auxiliary: Iterable[S | F | NPyConstantSeries | tuple[str, np.ndarray | float]] = (),
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
    def solutions(self) -> list[S| NPyConstantSeries]:
        return list(self._solutions)
    
    @property
    def auxiliary(self) -> list[S| NPyConstantSeries | NPyConstant]:
        return self._auxiliary

    @property
    def namespace(self) -> dict[str, F | NPyConstantSeries | float]:
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
        convert_func: Callable[[FunctionSeries], F],
        slc_const: slice = slice(None, None, None),
        auxiliary: bool = False, 
    ):
        _solutions = []
        for s in sim.solutions:
            if isinstance(s, ConstantSeries):
                _sltn = as_npy_constant_series(s, slc_const)
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
   

class GridSimulation(NPySimulation[GridMesh, GridFunction, GridFunctionSeries]):
    @classmethod
    def from_simulation(
        cls: type['GridSimulation'],
        sim: Simulation,
        slc_func: slice = slice(None, None, None),
        slc_const: slice = slice(None, None, None),
        auxiliary: bool = False,
        strict: bool = False,
        jit: bool = True,
        mask: float = np.nan,
        use_mesh_map: bool = False,
        use_mesh_cache: bool = True,
        use_func_cache: bool = True,
    ) -> Self:
        convert_func = lambda u: (
            as_grid_function_series(
                u, slc_func, strict, jit, mask, use_mesh_map, use_mesh_cache, use_func_cache,
            )
        )
        return super().from_simulation(
            sim,
            convert_func,
            slc_const,
            auxiliary,
        )


class TriSimulation(NPySimulation[TriMesh, TriFunction, TriFunctionSeries]):
    @classmethod
    def from_simulation(
        cls: type['TriSimulation'],
        sim: Simulation,
        slc_func: slice = slice(None, None, None),
        slc_const: slice = slice(None, None, None),
        auxiliary: bool = False,
    ) -> Self:
        convert_func = lambda u: (
            as_tri_function_series(u, ...)
        )
        return super().from_simulation(
            sim,
            convert_func,
            slc_const,
            auxiliary,
        )


@replicate_callable(GridSimulation.from_simulation)
def as_grid_simulation():
    pass


@replicate_callable(TriSimulation.from_simulation)
def as_tri_simulation():
    pass


def as_npy_simulation():
    ...