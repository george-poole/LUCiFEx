from typing import Callable

from dolfinx.mesh import Mesh, locate_entities, refine as dolfinx_refine

from ..utils.fenicsx_utils import (
    as_boolean_marker, is_simplicial, NonSimplexMeshError,
    BooleanMarker, MarkerAlias,
)


def refine(
    mesh: Mesh,
    marker: BooleanMarker | MarkerAlias,
    n_stop: int = 1,
    condition: Callable[[Mesh], bool] = None,
    redistribute: bool = True,
    name: str | None = None,
) -> Mesh:
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
        mesh_refined = dolfinx_refine(mesh_refined, facets, redistribute)
        _n += 1

    if name is not None:
        mesh_refined.name = name

    return mesh_refined


def copy_mesh(
    mesh: Mesh, 
    name: str | None = None,
) -> Mesh:
    # TODO dolfinx0.7.0+ to copy quadrilateral meshes
    if not is_simplicial(mesh):
        raise NonSimplexMeshError('Copying')
    nothing_marker = lambda x: (x[0] > 0) & (x[0] < 0) 
    mesh_copied = refine(mesh, nothing_marker, name=name)
    return mesh_copied