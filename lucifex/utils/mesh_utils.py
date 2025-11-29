from typing import Iterable

import numpy as np
from dolfinx.cpp.mesh import entities_to_geometry
from dolfinx.mesh import Mesh
from ufl.geometry import (GeometricCellQuantity, Circumradius, 
                          CellDiameter, MaxCellEdgeLength, MinCellEdgeLength)

from .dofs_utils import dofs
from .py_utils import optional_lru_cache


def mesh_vertices(
    mesh: Mesh,
    dim: int = None,
) -> list[tuple[float, ...]]:
    """`[(x₀, y₀, z₀), (x₁, y₁, z₁), (x₂, y₂, z₂), ...]`"""
    if dim is None:
        dim = mesh.geometry.dim
    vertices = [tuple(i[:dim]) for i in mesh.geometry.x]
    return vertices


def mesh_coordinates(
    mesh: Mesh,
    dim: int | None = None,
) -> tuple[np.ndarray, ...]:
    """
    `([x₀, x₁, x₂, ...], [y₀, y₁, y₂, ...], [z₀, z₁, z₂, ...])`
    """
    if dim is None:
        dim = mesh.geometry.dim
    return tuple(mesh.geometry.x[:, i] for i in range(dim))


@optional_lru_cache
def mesh_axes(
    mesh: Mesh,
    strict: bool = False,
)-> tuple[np.ndarray, ...]:
    """
    Unique and ordered `([x₀, x₁, x₂, ...], [y₀, y₁, y₂, ...], [z₀, z₁, z₂, ...])`
    """
    if strict and not is_cartesian(mesh):
        raise CartesianMeshError()
    return tuple(np.sort(np.unique(i)) for i in mesh_coordinates(mesh))


def mesh_vertices_tensor(
    mesh: Mesh,
    strict: bool = False,
) -> np.ndarray:
    """
    Tensor `v` such that `v[i, j]` returns the `(i, j)`th vertex of a structuted mesh. 
    """
    if strict and not is_cartesian(mesh):
        raise CartesianMeshError()
    mesh_axes = mesh_axes(mesh)
    if len(mesh_axes) == 2:
        x, y = mesh_axes
        return np.array([[(i, j) for j in y] for i in x])
    elif len(mesh_axes) == 3:
        x, y, z = mesh_axes
        return np.array([[[(i, j, k) for k in z] for j in y] for i in x])
    else:
        raise ValueError
    

def mesh_axes_spacing(
    mesh: Mesh,
    strict: bool = False,
) -> tuple[np.ndarray, ...]:
    """`([dx₀, dx₁, dx₂, ...], [dy₀, dy₁, dy₂, ...], [dz₀, dz₁, dz₂, ...])`"""
    if strict and not is_cartesian(mesh):
        raise CartesianMeshError()
    return tuple(np.diff(i) for i in mesh_axes(mesh))


@optional_lru_cache
def is_cartesian(
    mesh: Mesh,
) -> bool:
    axes = mesh_axes(mesh, strict=False)
    n_vertices = len(mesh.geometry.x)
    n_axes = [len(i) for i in axes]
    n_vertices_cartesian = np.prod(n_axes)
    return bool(n_vertices == n_vertices_cartesian)


@optional_lru_cache
def is_uniform_cartesian(
    mesh: Mesh,
) -> bool:
    if not is_cartesian(mesh):
        return False
    return all(all(np.isclose(dx[0], dx)) for dx in mesh_axes_spacing(mesh))


def n_entities(
    mesh: Mesh,
    dim: int,
    glob: bool = True,
    ghosts: bool = True,
) -> int:
    index_map = mesh.topology.index_map(dim)
    if glob:
        return index_map.size_global
    else:
        n = index_map.size_local
        if ghosts:
            n += index_map.num_ghosts
    return n


def n_cells(
    mesh: Mesh, 
    glob: bool = True,
    ghosts: bool = True,
) -> int:
    return n_entities(mesh, mesh.topology.dim, glob, ghosts)


def entities_to_coordinates(
    entity_indices: Iterable[int], 
    edim: int, 
    msh: Mesh, 
    centroid: bool = True,
) -> list[np.ndarray]:
    entity_indices = np.array(entity_indices, dtype=np.int32)
    vertex_indices = entities_to_geometry(msh, edim, entity_indices, False)

    coordinates = []
    for i in vertex_indices:
        vertex_coords = msh.geometry.x[i]
        if centroid:
            vertex_coords = np.mean(vertex_coords, axis=0)
        coordinates.append(vertex_coords)
    return coordinates


def cell_sizes(
    mesh: Mesh,
    h: str,
) -> np.ndarray:
    h = cell_size_quantity(mesh, h)
    return dofs(h, (mesh, 'DP', 0), use_cache=True)


def cell_aspect_ratios(mesh) -> np.ndarray:
    r = MaxCellEdgeLength(mesh) / MinCellEdgeLength(mesh)
    return dofs(r, (mesh, 'DP', 0), use_cache=True)


def cell_size_quantity(mesh: Mesh, h: str) -> GeometricCellQuantity:
    d = {
        'hmin': MinCellEdgeLength,
        'hmax': MaxCellEdgeLength,
        'hdiam': CellDiameter,
        'hrad': Circumradius,
    }
    try:
        return d[h](mesh)
    except KeyError:
        raise ValueError(f'Invalid cell size quantity {h}.')
    

def CartesianMeshError():
    return ValueError("Mesh vertices must form a Cartesian grid.")
