from abc import abstractmethod
from typing import TypeVar, Iterable, Generic, Callable
from typing_extensions import Self

import numpy as np
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr

from ..mesh.mesh2npy import NPyMesh, GridMesh, TriMesh, as_npy_object, QuadMesh
from ..fem import Function, Constant
from ..fem.fem2npy import (
    NPyFunction, GridFunction, TriFunction, NPyConstant, QuadFunction,
    as_grid_function, as_tri_function, as_npy_constant,
)
from ..fdm import FunctionSeries, ConstantSeries, ExprSeries
from ..fdm.fdm2npy import (
    NPyFunctionSeries, GridFunctionSeries, QuadFunctionSeries, as_grid_function_series,
    as_tri_function_series, TriFunctionSeries, NumericSeries, as_npy_constant_series,
)
from ..utils.py_utils import MultiKey, MultipleDispatchTypeError, replicate_callable, StrSlice, as_slice
from .simulation import Simulation


M = TypeVar('M', bound=NPyMesh)
F = TypeVar('F', bound=NPyFunction)
S = TypeVar('S', bound=NPyFunctionSeries)
class NPySimulation(
    Generic[M, F, S],
    MultiKey[str, S | F | NumericSeries | np.ndarray | float]
):
    def __init__(
        self,
        solutions: Iterable[S | NumericSeries],
        auxiliary: Iterable[S | F | NumericSeries | NPyConstant | tuple[str, float | int | np.ndarray]] = (),
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
    def solutions(self) -> list[S| NumericSeries]:
        return list(self._solutions)
    
    @property
    def auxiliary(self) -> list[S| NumericSeries | NPyConstant]:
        return self._auxiliary

    @property
    def namespace(self) -> dict[str, F | NumericSeries | float]:
        d = {i.name: i for i in self._solutions}
        d.update({f.name: f for f in self._auxiliary if not isinstance(f, tuple)})        
        d.update({f[0]: f[1] for f in self._auxiliary if isinstance(f, tuple)})
        return d
    
    @property
    def meshes(self) -> list[Mesh]:
        return [i.mesh for i in self.solutions if isinstance(i, NPyFunctionSeries)]
    
    @property
    def mesh(self) -> M | None:
        if len(set(self.meshes)) == 1:
            return self.meshes[0]
        else:
            return None
    
    @property
    def timings(self) -> dict | None:
        return self._timings

    @classmethod
    @abstractmethod
    def from_simulation(
        cls,
        sim: Simulation,
        convert_series: Callable[[FunctionSeries | ExprSeries], S],
        convert_func: Callable[[Function | Expr], F],
        slc_const: StrSlice = ':',
        auxiliary: bool = True, 
    ):
        _solutions = []
        for s in sim.solutions:
            if isinstance(s, ConstantSeries):
                _sltn = as_npy_constant_series(s, as_slice(slc_const))
            else:
                _sltn = convert_series(s)
            _solutions.append(_sltn)

        _as_npy = lambda arg: (
            as_npy_or_numeric(arg, convert_series, convert_func)
        )

        _auxiliary = []
        if auxiliary:
            for aux in sim.auxiliary:
                if isinstance(aux, tuple):
                    name, value = aux
                    _aux = (name, _as_npy(value))
                else:
                    _aux = _as_npy(aux)
                _auxiliary.append(_aux)

        return cls(_solutions, _auxiliary, sim.timings)
    

F = TypeVar('F', bound=NPyFunction)
S = TypeVar('S', bound=NPyFunctionSeries)
def as_npy_or_numeric(
    obj,
    convert_series: Callable[[FunctionSeries | ExprSeries], S],
    convert_func: Callable[[Function | Expr], F],
):
    if isinstance(obj, (float, int, np.ndarray)):
        return obj
    if isinstance(obj, Constant):
        return as_npy_constant(obj)
    if isinstance(obj, (Function, Expr)):
        return convert_func(obj)
    if isinstance(obj, ExprSeries):
        return convert_series(obj)
    raise MultipleDispatchTypeError(obj)
   

class GridSimulation(NPySimulation[GridMesh, GridFunction, GridFunctionSeries]):
    @classmethod
    def from_simulation(
        cls: type['GridSimulation'],
        sim: Simulation,
        slc_func: StrSlice = ':',
        slc_const: StrSlice = ':',
        auxiliary: bool = True,
        strict: bool = False,
        jit: bool = True,
        mask: float = np.nan,
        use_mesh_map: bool = True,
        use_mesh_cache: bool = True,
        use_func_cache: bool = True,
    ) -> Self:
        convert_series = lambda u: (
            as_grid_function_series(
                u, slc_func, strict, jit, mask, use_mesh_map, use_mesh_cache, use_func_cache, sim.mesh,
            )
        )
        convert_func = lambda u: (
            as_grid_function(u, strict, jit, mask, use_mesh_map, use_mesh_cache, sim.mesh)
        )
        return super().from_simulation(
            sim,
            convert_series,
            convert_func,
            slc_const,
            auxiliary,
        )


class TriSimulation(NPySimulation[TriMesh, TriFunction, TriFunctionSeries]):
    @classmethod
    def from_simulation(
        cls: type['TriSimulation'],
        sim: Simulation,
        slc_func: StrSlice = ':',
        slc_const: StrSlice = ':',
        auxiliary: bool = True,
        use_mesh_cache: bool = True,
        use_func_cache: bool = True,
    ) -> Self:
        convert_series = lambda u: (
            as_tri_function_series(u, slc_func, use_mesh_cache, use_func_cache, sim.mesh),
        )
        convert_func = lambda u: (
            as_tri_function(u, use_mesh_cache, sim.mesh),
        )
        return super().from_simulation(
            sim,
            convert_series,
            convert_func,
            slc_const,
            auxiliary,
        )
    

class QuadSimulation(NPySimulation[QuadMesh, QuadFunction, QuadFunctionSeries]):
    @classmethod
    def from_simulation(
        cls: type['QuadSimulation'],
        sim: Simulation,
        slc_func: StrSlice = ':',
        slc_const: StrSlice = ':',
    ) -> Self:
        raise NotImplementedError
    

@replicate_callable(GridSimulation.from_simulation)
def as_grid_simulation():
    pass


@replicate_callable(TriSimulation.from_simulation)
def as_tri_simulation():
    pass


@replicate_callable(QuadSimulation.from_simulation)
def as_quad_simulation():
    pass


def as_npy_simulation(
    sim: Simulation,
    grid: bool | None = None,
    use_mesh_cache: bool = True,  
):
    return as_npy_object(
        sim,
        as_grid_simulation,
        as_tri_simulation,
        as_quad_simulation,
        sim.mesh,
        grid,
        use_mesh_cache,
    )