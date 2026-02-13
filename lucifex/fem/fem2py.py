from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self

import numpy as np
import numba
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh

from . import Function, Constant
from ..fdm import Series, ExprSeries, ConstantSeries
from ..sim import Simulation
from ..utils.py_utils import optional_lru_cache, replicate_callable
from ..utils.fenicsx_utils import (
    create_function, dofs,
    mesh_vertices, mesh_axes,
    is_scalar, NonScalarError,
)


class FE2PyMesh(ABC):
    @abstractmethod
    def __init__(
        self, 
        name: str | None = None,
    ):
        self._name = name

    @property
    def name(self) -> str | None:
        return self._name


M = TypeVar('M', bound=FE2PyMesh)
class FE2PyFunction(Generic[M]):
    def __init__(
        self,
        values: np.ndarray,
        mesh: M,
        name: str | None = None,
    ):
        self._values = values
        self._mesh = mesh
        self._name = name

    @property
    def values(self) -> np.ndarray: #FIXME shape type hints
        return self._values
    
    @property 
    def shape(self) -> tuple[int, ...]:
        return self.values.shape
    
    @property
    def mesh(self) -> M:
        return self._mesh
    
    @classmethod
    @abstractmethod
    def from_function(
        cls,
        u: Function,
        values_func: Callable[[Function], np.ndarray],
        mesh_func: Callable[[Mesh], M],
    ):
        values = values_func(u)
        msh = mesh_func(u.function_space.mesh)
        return cls(
            values,
            msh,
            u.name,
        )
    

class GridMesh(FE2PyMesh):
    def __init__(
        self, 
        axes: tuple[np.ndarray, ...],
        name: str | None = None,
    ):
        super().__init__(name)
        self._axes = axes

    @classmethod
    def from_mesh(
        cls: type['GridMesh'], 
        mesh: Mesh,
        strict: bool = False,
    ) -> Self:
        axes = mesh_axes(use_cache=True)(mesh, strict)
        name = mesh.name
        return cls(axes, name)
    
    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        return self._axes
    
    @property
    def x_axis(self) -> np.ndarray:
        return self._get_axis(0)
    
    @property
    def y_axis(self) -> np.ndarray | None:
        return self._get_axis(1)
    
    @property
    def z_axis(self) -> np.ndarray | None:
        return self._get_axis(2)
        
    def _get_axis(self, index: int):
        try:
            return self._axes[index]
        except IndexError:
            return None 
    

class TriMesh(FE2PyMesh):
    def __init__(
        self, 
        x_coordinates: np.ndarray, 
        y_coordinates: np.ndarray, 
        triangles: list[np.ndarray], # TODO type hint dtypeint and shape
        name: str | None = None,
    ):
        super().__init__(name)
        self._x_coordinates = x_coordinates
        self._y_coordinates = y_coordinates
        self._triangles = triangles
    
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
    
    @cached_property
    def triangulation(self) -> Triangulation:
        return Triangulation(
            self._x_coordinates,
            self._y_coordinates,
            self._triangles,
        )
    
    @property
    def x_coordinates(self):
        return self._x_coordinates

    @property
    def y_coordinates(self):
        return self._y_coordinates

    @property
    def triangles(self):
        return self._triangles
    

as_grid_mesh = optional_lru_cache(
    replicate_callable(GridMesh.from_mesh)(lambda: None)
)
as_tri_mesh = optional_lru_cache(
    replicate_callable(TriMesh.from_mesh)(lambda: None)
)


class GridFunction(FE2PyFunction[GridMesh]):
    @classmethod
    def from_function(
        cls: type['GridFunction'],
        u: Function,
        strict: bool = False,
        jit: bool = True,
        mask: float = np.nan,
        use_mesh_map: bool = False,
        use_mesh_cache: bool = True,
    ) -> Self:
        values_func = lambda u: grid_dofs(u, strict, jit, mask, use_mesh_map, use_mesh_cache)
        mesh_func = lambda m: as_grid_mesh(use_cache=use_mesh_cache)(m, strict)
        return super().from_function(
            u,
            values_func,
            mesh_func,
        )
    

class TriFunction(FE2PyFunction[TriMesh]):
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
        values_func = lambda u: dofs(u, ('P', 1), try_identity=True)
        mesh_func = lambda m: as_tri_mesh(use_cache=use_mesh_cache)(m)
        return super().from_function(
            u,
            values_func,
            mesh_func,
        )


as_grid_function = optional_lru_cache(
    replicate_callable(GridFunction.from_function)(lambda: None)
)
as_tri_function = optional_lru_cache(
    replicate_callable(TriFunction.from_function)(lambda: None)
)








def grid_dofs(
    f: Function | Expr,
    strict: bool = False,
    jit: bool = True,
    mask: float = np.nan,
    use_mesh_map: bool = False,
    use_mesh_cache: bool = True,
) -> np.ndarray:
    """Interpolates the finite element function to the `P₁` function space
    (which has identity vertex-to-DoF map) to evaluate the function at the vertex values,
    then returns these vertex values in a `np.ndarray`.

    Note that this is suitable only if the mesh is Cartesian.
    """
    if not is_scalar(f):
        raise NonScalarError(f)
    
    if not isinstance(f, Function):
        f = create_function((mesh, 'P', 1), f)
    
    mesh = f.function_space.mesh
    axes = mesh_axes(use_cache=use_mesh_cache)(mesh, strict)
    vertices = mesh_vertices(mesh)
    vertex_values = dofs(f, ('P', 1), try_identity=True)

    n_vertices = len(vertices)
    n_values = len(vertex_values)
    if n_vertices != n_values:
        raise ValueError(
            f""" 
        Number of vertex values {n_values} does not 
        match number of vertices {n_vertices} """
        )

    if use_mesh_map:
        mapping = vertex_to_grid_index_map(use_cache=use_mesh_cache)(
            mesh, strict, jit, use_mesh_cache
        )
        return _grid_dofs_from_map(mapping, vertex_values, axes, mask)
    else:
        if jit:
            return _grid_dofs_jit(vertex_values, numba.typed.List(vertices), axes, mask)
        else:
            return _grid_dofs_nojit(vertex_values, vertices, axes, mask)


@numba.jit(nopython=True)
def _grid_dofs_jit(
    vertex_values: np.ndarray,
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    """
    Returns the vertex values on a Cartesian grid.
    See also `_grid_nojit` for a slower, non-compiled version.
    """

    n_vertex = len(vertex_values)
    dim = len(axes)

    if dim ==1 :
        nx = len(axes[0])
        f_grid = np.full((nx, ), mask)
        for vertex_index in range(n_vertex):
            x_index = np.searchsorted(axes[0], vertices[vertex_index][0])
            f_grid[x_index] = vertex_values[vertex_index]

    elif dim == 2:
            nx, ny = len(axes[0]), len(axes[1])
            f_grid = np.full((nx, ny), mask)
            for vertex_index in range(n_vertex):
                x_index = np.searchsorted(axes[0], vertices[vertex_index][0])
                y_index = np.searchsorted(axes[1], vertices[vertex_index][1])
                f_grid[x_index, y_index] = vertex_values[vertex_index]
    elif dim == 3:
        nx, ny, nz = len(axes[0]), len(axes[1]), len(axes[2])
        f_grid = np.full((nx, ny, nz), mask)
        for vertex_index in range(n_vertex):
            x_index = np.searchsorted(axes[0], vertices[vertex_index][0])
            y_index = np.searchsorted(axes[1], vertices[vertex_index][1])
            z_index = np.searchsorted(axes[2], vertices[vertex_index][2])
            f_grid[x_index, y_index, z_index] = vertex_values[vertex_index]
    else:
        raise ValueError(f"Spatial dimension should be 1, 2 or 3, not {dim}")

    return f_grid


def _grid_dofs_nojit(
    vertex_values: np.ndarray,
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    """
    Returns the vertex values on a Cartesian grid. 
    See also `_grid_jit` for a faster, JIT-compiled version.
    """
    return _grid_dofs_jit.__wrapped__(vertex_values, vertices, axes, mask)


def _grid_dofs_from_map(
    vertex_grid_map: np.ndarray,
    vertex_values: np.ndarray,
    x_axes: tuple[np.ndarray, ...],
    mask: float,
) -> np.ndarray:
    nx = tuple(len(i) for i in x_axes)
    f_grid = np.full(nx, mask)

    for vertex_index, x_index in vertex_grid_map.items():
        f_grid[x_index] = vertex_values[vertex_index]

    return f_grid


@optional_lru_cache
def vertex_to_grid_index_map(
    mesh: Mesh,
    strict: bool = False,
    jit: bool = True,
    use_cache: bool = True,
) -> dict[int, tuple[int, ...]]:
    axes = mesh_axes(use_cache=use_cache)(mesh, strict)
    vertices = mesh_vertices(mesh)
    if jit:
        map_array = _vertex_to_grid_index_map_jit(numba.typed.List(vertices), axes)
    else:
        map_array = _vertex_to_grid_index_map_nojit(vertices, axes)
    return {i: tuple(map_array[i]) for i in range(len(map_array))}


@numba.jit(nopython=True)
def _vertex_to_grid_index_map_jit(
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    n_vertices = len(vertices)
    dim = len(axes)
    map_array = np.empty((n_vertices, dim), dtype=np.int64)

    for vertex_index in range(n_vertices):
        for d in range(dim):
            map_array[vertex_index, d] = np.searchsorted(axes[d], vertices[vertex_index][d])

    return map_array


def _vertex_to_grid_index_map_nojit(
    vertices: list[tuple[float, ...]],
    axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    return _vertex_to_grid_index_map_jit.__wrapped__(vertices, axes)






