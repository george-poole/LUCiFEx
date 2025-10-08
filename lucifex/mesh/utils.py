from typing import (
    Callable, ParamSpec,
    Iterable, TypeAlias, Any,
)
from functools import partial

import gmsh
from mpi4py import MPI
from dolfinx.mesh import Mesh, MeshTagsMetaClass
from dolfinx.cpp.mesh import MeshTags_int32
from dolfinx.io.gmshio import model_to_mesh

from ..utils import CellType


Markers: TypeAlias = Iterable[tuple[str, int]]
P = ParamSpec('P')
def create_gmsh_mesh_factory(
    model_func: Callable[P, gmsh.model],
    dim: int,
    default_name: str,
):  
    if dim == 2:
        default_cell = CellType.TRIANGLE
    if dim == 3:
        default_cell = CellType.TETRAHEDRON

    def _inner(
        h: float | tuple[float, float],
        cell: CellType = default_cell,
        name: str = default_name,
        comm = MPI.COMM_WORLD,
        rank: int = 0,
        markers: Iterable[tuple[str, int, Callable[[gmsh.model], tuple]]] = (('cells', 1, partial(get_entity_tags, dim=dim)), ),
        meshtags: bool = False,
        gmsh_set_number: dict | None = None,
        **gmsh_set_mesh,
    ) -> Callable[P, Mesh | tuple[Mesh, MeshTags_int32, MeshTagsMetaClass]]:    
        def _(*args: P.args, **kwargs: P.kwargs):
            if comm.rank == rank:
                initialize_model(gmsh.model, name)
                model = model_func(*args, **kwargs)
                gmsh.model.occ.synchronize()
                gmsh.model.geo.synchronize()
                set_model_groups(gmsh.model, markers)
                set_model_options(h, cell, gmsh_set_number, gmsh_set_mesh)
                gmsh.model.occ.synchronize()
                gmsh.model.geo.synchronize()
                gmsh.model.mesh.generate(dim)
                mesh, cell_meshtags, facet_meshtags = model_to_mesh(model, comm, rank, dim)
                gmsh.finalize()
                mesh.name = name
                if meshtags:
                    return mesh, cell_meshtags, facet_meshtags  
                else:
                    return mesh
        return _
    
    return _inner


def get_entity_tags(
    model: gmsh.model,
    dim: int,
) -> tuple[int, list[int]]:
    cells = model.getEntities(dim)  
    return cells[0][0], [cells[0][1]] 
    

def initialize_model(
    model: gmsh.model,
    name: str,
) -> None:
    if not gmsh.isInitialized():
        gmsh.initialize()
    model.add(name)


def set_model_groups(
    model: gmsh.model,
    markers: Iterable[tuple[str, int, Callable]],
) -> None:
    for name, tag, get_tags in markers:
        dim, tags = get_tags(model)
        model.addPhysicalGroup(
            dim, tags, tag, name,
        )


def set_model_options(
    h: float | tuple[float, float],  
    cell: str,
    gmsh_set_number: dict[str, Any], 
    gmsh_set_mesh: dict[str, Any],
) -> None:
    if cell in (CellType.HEXAHEDRON,  CellType.QUADRILATERAL):
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

    if not isinstance(h, tuple):
        h = (h, h)

    _gmsh_set_mesh = {
        "CharacteristicLengthMin": h[0],
        "CharacteristicLengthMax": h[1],
    }
    _gmsh_set_mesh.update(gmsh_set_mesh)
    for name, value in _gmsh_set_mesh.items():
        gmsh.option.setNumber(f'Mesh.{name}', value)

    if gmsh_set_number is None:
        gmsh_set_number = {}
    _gmsh_set_number = {"General.Verbosity": 0}
    _gmsh_set_number.update(gmsh_set_number)
    for name, value in _gmsh_set_number.items():
        gmsh.option.setNumber(name, value)
