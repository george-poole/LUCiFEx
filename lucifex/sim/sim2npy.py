from abc import ABC, abstractmethod
from typing import TypeVar, Iterable, Generic, Callable
from typing_extensions import Self

from dolfinx.fem import Function
from ufl.core.expr import Expr
import numpy as np

from ..mesh.mesh2npy import NPyMesh, GridMesh, TriMesh
from ..fem import Function, Constant
from ..fem.fem2npy import NPyFunction, GridFunction, TriFunction
from ..fdm import Series, FunctionSeries, ConstantSeries, ExprSeries
from ..fdm.fdm2npy import NPySeries, GridSeries, TriSeries, FloatSeries
from ..utils.py_utils import MultiKey
from . import Simulation
from ..utils.py_utils import replicate_callable


M = TypeVar('M', bound=NPyMesh)
F = TypeVar('F', bound=NPyFunction)
S = TypeVar('S', bound=NPySeries)
class NPySimulation(
    Generic[M, F, S],
    MultiKey[str, S | F | FloatSeries | np.ndarray | float]
):
    def __init__(
        self,
        solutions: Iterable[S | FloatSeries],
        auxiliary: Iterable[S | F | FloatSeries | tuple[str, np.ndarray | float]] = (),
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
    def solutions(self) -> list[S| FloatSeries]:
        return list(self._solutions)
    
    @property
    def auxiliary(self) -> list[S| FloatSeries | tuple[str, np.ndarray | float]]:
        return self._auxiliary

    @property
    def namespace(self) -> dict[str, F | FloatSeries | float]:
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
        convert_func: Callable[[Series], F],
        auxiliary: bool = False, 
    ):
        _solutions = []
        for s in sim.solutions:
            if isinstance(s, ConstantSeries):
                _sltn = ...
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
   

class GridSimulation(NPySimulation[GridMesh, GridFunction, GridSeries]):
    @classmethod
    def from_simulation(
        cls: type['GridSimulation'],
        sim: Simulation,
        slc: slice = slice(None, None, None),
        auxiliary: bool = False,
        strict: bool = False,
        jit: bool = True,
        mask: float = np.nan,
        use_mesh_map: bool = False,
        use_mesh_cache: bool = True,
        use_func_cache: bool = True,
    ) -> Self:
        convert_func = lambda u: (
            GridSeries.from_series(
                u, slc, strict, jit, mask, use_mesh_map, use_mesh_cache, use_func_cache,
            )
        )
        return super().from_simulation(
            sim,
            convert_func,
            auxiliary,
        )


class TriSimulation(NPySimulation[TriMesh, TriFunction, TriSeries]):
    ...


@replicate_callable(GridSimulation.from_simulation)
def as_grid_simulation():
    pass


@replicate_callable(TriSimulation.from_simulation)
def as_tri_simulation():
    pass