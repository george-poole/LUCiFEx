from typing_extensions import Self

from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
import numba
import numpy as np

from ..fdm import FunctionSeries
from ..utils.fenicsx_utils import (
    create_function, dofs,
    mesh_vertices, mesh_axes,
    is_scalar, NonScalarError,
)
from ..sim import Simulation
from ..utils.py_utils import optional_lru_cache, replicate_callable
from .abc import FE2PyMesh, FE2PyFunction, FE2PySeries, FE2PySimulation


class GridSimulation(FE2PySimulation[GridMesh, GridFunction, GridSeries]):
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


@replicate_callable(GridSimulation.from_simulation)
def as_grid_simulation():
    pass

     