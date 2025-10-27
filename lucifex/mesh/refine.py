from typing import Callable, Iterable

from dolfinx.mesh import Mesh, locate_entities, refine as dolfinx_refine

from ..utils import as_spatial_marker, MarkerOrExpression
from .cartesian import CellType


def refine(
    mesh: Mesh,
    marker: MarkerOrExpression | Iterable[MarkerOrExpression],
    n_stop: int = 1,
    condition: Callable[[Mesh], bool] = None,
    redistribute: bool = True,
    name: str | None = None,
) -> Mesh:
    """
    For simplex meshes only.
    """
    if not is_simplex_mesh(mesh):
        raise ValueError('Only implemented for simplex meshes.')
    
    marker = as_spatial_marker(marker)

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


def is_simplex_mesh(mesh: Mesh) -> bool:
    dim = mesh.geometry.dim
    cell_name = mesh.topology.cell_name()

    match dim:
        case 1:
            return True
        case 2:
            return cell_name == CellType.TRIANGLE
        case 3:
            return cell_name == CellType.TETRAHEDRON


def copy_mesh(
    mesh: Mesh, 
    name: str | None = None,
) -> Mesh:
    """
    For simplex meshes only
    """
    # TODO dolfinx 0.7.0+ features to handle non-simplex meshes
    nothing_marker = lambda x: (x[0] > 0) & (x[0] < 0) 
    mesh_copied = refine(mesh, nothing_marker, name=name)
    return mesh_copied