from functools import cached_property
from typing_extensions import Self

from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from matplotlib.tri.triangulation import Triangulation
import numpy as np

from ..fdm import FunctionSeries
from ..utils.fenicsx_utils import (
    dofs, mesh_coordinates, is_simplicial,
)
from ..utils.py_utils import optional_lru_cache, replicate_callable
from .abc import FE2PyMesh, FE2PyFunction, FE2PySeries, FE2PySimulation


class TriSeries(
    FE2PySeries[TriMesh, TriFunction]
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
        cls: type['TriSeries'], 
        u: FunctionSeries,
        use_cache: tuple[bool, bool] = (True, True),
        slc: slice = slice(None, None, None),
        **tri_kwargs,
    ) -> Self:
        series, time_series, trigl = super().from_series(
            trigl,
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
        return TriSeries([i[index] for i in self.series], self.time_series, self.triangulation, name)
    


class TriSimulation(FE2PySimulation[TriSeries]):
    ...

    




# @singledispatch
# def _triangulation(arg, *_, **__):
#     raise MultipleDispatchTypeError(arg)

# @_triangulation.register
# def _(mesh: Mesh):
    # if not mesh.geometry.dim == 2:
    #     raise ValueError(
    #         f"""
    #     Triangulation only valid in 2D, not 
    #     dimension {mesh.geometry.dim} """
    #     )
    
    # if not is_simplicial(mesh):
    #     raise ValueError(
    #         f""" 
    #     Triangulation only valid for triangle cells, not 
    #     {mesh.topology.cell_name()} """
    #     )

    # x_coordinates, y_coordinates = mesh_coordinates(mesh)

    # # coordinates of each vertex in a given triangle cell
    # connec = mesh.topology.connectivity(mesh.geometry.dim, 0)
    # n_cells = connec.num_nodes
    # triangles = [connec.links(i) for i in range(n_cells)]

    # trigl = Triangulation(x_coordinates, y_coordinates, triangles)

    # return trigl


# triangulation = optional_lru_cache(_triangulation)