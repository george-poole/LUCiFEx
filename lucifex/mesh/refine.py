from typing import Callable

import numpy as np
from dolfinx.mesh import Mesh, locate_entities, refine, create_mesh

from ..utils.fenicsx_utils import (
    as_boolean_marker, is_simplicial, NonSimplexMeshError,
    BooleanMarker, MarkerAlias,
)


def refine_mesh(
    mesh: Mesh,
    marker: BooleanMarker | MarkerAlias,
    n_stop: int = 1,
    condition: Callable[[Mesh], bool] = None,
    redistribute: bool = True,
    name: str | None = None,
) -> Mesh:
    """
    For simplicial meshes only.
    """
    if not is_simplicial(mesh):
        raise NonSimplexMeshError('Refinement')
    
    marker = as_boolean_marker(marker)

    if condition is None:
        condition = lambda _: False

    mesh_refined = mesh
    _n = 0
    while _n < n_stop and not condition(mesh_refined):
        fdim = mesh.topology.dim - 1
        mesh_refined.topology.create_entities(fdim)
        facets = locate_entities(mesh_refined, fdim, marker)
        mesh_refined = refine(mesh_refined, facets, redistribute)
        _n += 1

    if name is not None:
        mesh_refined.name = name

    return mesh_refined


def copy_mesh(
    mesh: Mesh, 
    name: str | None = None,
) -> Mesh:
    
    if is_simplicial(mesh):
        nothing_marker = lambda x: (x[0] > 0) & (x[0] < 0) 
        return refine_mesh(mesh, nothing_marker, name=name)
    else:
        gdim = mesh.geometry.dim 
        tdim = mesh.topology.dim
        x = np.copy(mesh.geometry.x[:, :gdim])
        mesh.topology.create_connectivity(tdim, 0)
        connec = mesh.topology.connectivity(tdim, 0)
        cells =  np.copy(connec.array.reshape(-1, 4)).astype(np.int64)
        ufl_domain = mesh.ufl_domain()
        mesh_copied =  create_mesh(mesh.comm, cells, x, ufl_domain)
        if name is not None:
            mesh_copied.name = name
        return mesh_copied
    
