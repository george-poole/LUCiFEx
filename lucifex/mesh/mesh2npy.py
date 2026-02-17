from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal
from typing_extensions import Self

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.tri.triangulation import Triangulation
from dolfinx.mesh import Mesh

from ..utils.py_utils import optional_lru_cache, replicate_callable
from ..utils.fenicsx_utils import (
    is_simplicial, mesh_coordinates,
    mesh_axes,
)


class NPyMesh(ABC):
    @abstractmethod
    def __init__(
        self, 
        name: str | None = None,
    ):
        self._name = name

    @property
    def name(self) -> str | None:
        return self._name
    

class GridMesh(NPyMesh):
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
    

class TriMesh(NPyMesh):
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
    

class QuadMesh(NPyMesh):
    def __init__(
        self, 
        x_coordinates, 
        y_coordinates, 
        quads,
        name: str | None = None,
    ):
        super().__init__(name)
        raise NotImplementedError

    @classmethod
    def from_mesh(
        cls: type['QuadMesh'], 
        mesh: Mesh,
        **polycollection_kwargs,
    ) -> Self:
        if not mesh.geometry.dim == 2:
            raise ValueError(
                f"""
            Quadrangulation only valid in 2D, not 
            dimension {mesh.geometry.dim} """
            )

        if is_simplicial(mesh):
            raise ValueError(
                f""" 
            Quadrangulation only valid with quadrilateral cells, not 
            {mesh.topology.cell_name()} """
            )

        xy_coordinates = mesh.geometry.x[:, :2]

        connec = mesh.topology.connectivity(mesh.geometry.dim, 0)
        n_cells = connec.num_nodes
        quads = [_reorder_quad_vertices(connec.links(i)) for i in range(n_cells)]

        vertices = xy_coordinates[quads]

        _kwargs = dict(facecolor='white', edgecolor='black')
        _kwargs.update(polycollection_kwargs)
        quadl = PolyCollection(vertices, **_kwargs)

        raise NotImplementedError
    
    @cached_property
    def triangulation(self) -> Triangulation:
        return Triangulation(
            self._x_coordinates,
            self._y_coordinates,
            self._triangles,
        )


as_grid_mesh = optional_lru_cache(
    replicate_callable(GridMesh.from_mesh)(lambda: None)
)
as_tri_mesh = optional_lru_cache(
    replicate_callable(TriMesh.from_mesh)(lambda: None)
)
as_quad_mesh = optional_lru_cache(
    replicate_callable(QuadMesh.from_mesh)(lambda: None)
)


def _reorder_quad_vertices(
    cell_vertices: np.ndarray,
    initial_ordering: Literal["N", "Z"] = "N",
    final_ordering: Literal["clockwise", "anticlockwise"] = "clockwise",
) -> np.ndarray:
    """
    Quadrilateral meshes generated by `dolfinx` have a local
    indexing of cell vertices that is self-intersecting 
    (e.g. forming an N-shape or Z-shape)

    ```
    1 _______ 3
    |         |
    |         |
    0 ‾‾‾‾‾‾‾ 2
    ```

    A `matplotlib` quadrangulation requires local indexing
    of cell vertices to be non-self-intersecting (i.e. clockwise
    or anticlockwise)

    ```
    1 _______ 2
    |         |
    |         |
    0 ‾‾‾‾‾‾‾ 3
    ```

    """
    assert len(cell_vertices) == 4

    if initial_ordering == 'N':
        if final_ordering == "clockwise":
            initial_final_map = {
                0: 0, 
                1: 1,
                2: 3,
                3: 2,
            }
        elif final_ordering == "anticlockwise":
            initial_final_map = {
                0: 0, 
                1: 2,
                2: 3,
                3: 1,
            }
        else:
            raise ValueError(f'{final_ordering} not recognised')
        
    elif initial_ordering == 'Z':
        raise NotImplementedError
    
    else:
        raise ValueError(f'{initial_ordering} not recognised')
    
    reordered = np.zeros(len(cell_vertices), dtype=np.int32)
    for initial, final in initial_final_map.items():
        reordered[initial] = cell_vertices[final]
    
    return reordered





