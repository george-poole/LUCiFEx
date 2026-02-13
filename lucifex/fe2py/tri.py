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
from .abc import FE2PySeries


class TriMesh:
    def __init__(
        self, 
        x_coordinates: np.ndarray, 
        y_coordinates: np.ndarray, 
        triangles: list[np.ndarray], # TODO type int and shape
        name: str | None = None,
    ):
        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates
        self._triangles = triangles
        self._name = name
    
    @classmethod
    def from_mesh(
        cls: type['TriMesh'], 
        mesh: Mesh,
    ) -> Self:
        if not mesh.geometry.dim == 2:
            raise ValueError(
                f"""
            Triangulation only valid in 2D, not 
            dimension {mesh.geometry.dim} """
            )
        
        if not is_simplicial(mesh):
            raise ValueError(
                f""" 
            Triangulation only valid for triangle cells, not 
            {mesh.topology.cell_name()} """
            )

        x_coordinates, y_coordinates = mesh_coordinates(mesh)

        connec = mesh.topology.connectivity(mesh.geometry.dim, 0)
        n_cells = connec.num_nodes
        triangles = [connec.links(i) for i in range(n_cells)]

        return cls(x_coordinates, y_coordinates, triangles, mesh.name)
    
    @property
    def name(self) -> str | None:
        return self._name
    
    @cached_property
    def triangulation(self) -> Triangulation:
        return Triangulation(
            self._x_coordinates,
            self._y_coordinates,
            self._triangles,
        )
    
    @property
    def x(self):
        return self._x_coordinates

    @property
    def y(self):
        return self._y_coordinates

    @property
    def triangles(self):
        return self._triangles
    

as_tri_mesh = optional_lru_cache(replicate_callable(TriMesh.from_mesh)(lambda: None))


class TriFunction:

    def __init__(
        self, 
        values: np.ndarray,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        triangles: list,
        name: str | None = None,
    ):
        self._values = values
        self._tri = TriMesh(x_coordinates, y_coordinates, triangles)
        self._name = name

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def tri(self) -> TriMesh:
        return self._tri
    
    @property
    def name(self) -> str | None:
        return self._name

    @classmethod
    def from_function(
        cls: type['TriFunction'],
        u: Function,
        use_mesh_cache: bool = True,
    ) -> Self:
        """
        Interpolates function to P₁ (which has identity vertex-to-DoF map)
        to evaluate the function at the vertex values.

        Note that this is suitable on both Cartesian and unstructured meshes.
        """
        values = dofs(u, ('P', 1), try_identity=True)
        tri_mesh = as_tri_mesh(use_cache=use_mesh_cache)(u.function_space.mesh)
        return cls(values, tri_mesh.x, tri_mesh.y, tri_mesh.triangles)


as_tri_function = optional_lru_cache(replicate_callable(TriFunction.from_function)(lambda: None))


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