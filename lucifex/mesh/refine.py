from typing import Callable, Iterable

from dolfinx.mesh import Mesh, locate_entities, refine as dolfinx_refine

from ..utils import as_spatial_indicator_func, SpatialMarkerFunc
from .cartesian import CellType
from .utils import overload_mesh, copy_mesh


@overload_mesh
def refine(
    mesh: Mesh,
    marker: SpatialMarkerFunc | Iterable[SpatialMarkerFunc],
    n_stop: int = 1,
    condition: Callable[[Mesh], bool] = None,
    redistribute: bool = True,
) -> None:
    assert is_simplex_mesh(mesh)
    marker = as_spatial_indicator_func(marker)

    if condition is None:
        condition = lambda _: False

    mesh_refined = copy_mesh(mesh)
    _n = 0
    while _n < n_stop and not condition(mesh_refined):
        fdim = mesh.topology.dim - 1
        mesh_refined.topology.create_entities(fdim)
        facets = locate_entities(mesh_refined, fdim, marker)
        mesh_refined = dolfinx_refine(mesh_refined, facets, redistribute)
        _n += 1

    mesh.__init__(
        mesh_refined.comm,
        mesh_refined.topology,
        mesh_refined.geometry,
        mesh_refined.ufl_domain(),
    )


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
