from typing import Iterable
from functools import partial

import gmsh
from mpi4py import MPI
from dolfinx.mesh import Mesh, MeshTagsMetaClass
from dolfinx.cpp.mesh import MeshTags_int32
from dolfinx.io.gmshio import model_to_mesh

from ..utils.enum_types import CellType


def get_entity_tags(
    model: gmsh.model,
    dim: int,
) -> tuple[int, list[int]]:
    cells = model.getEntities(dim)  
    return cells[0][0], [cells[0][1]], 
    
    
get_cell_tags = partial(get_entity_tags, dim=2)


def _model_to_mesh(
    model: gmsh.model,
    comm: MPI.Comm,
    rank: int,
    dim: int,
    meshtags: bool,
) -> Mesh | tuple[Mesh, MeshTags_int32, MeshTagsMetaClass]:
    mesh, cell_meshtags, facet_meshtags = model_to_mesh(model, comm, rank, dim)
    gmsh.finalize()
    if meshtags:
        return mesh, cell_meshtags, facet_meshtags  
    else:
        return mesh


def annulus_mesh(
    centre: tuple[float, float],
    Rinner: float,
    Router: float, 
    Nr: int,
    name: str,
    cell: CellType = CellType.TRIANGLE,
    structured: bool = False, #TODO
    comm = MPI.COMM_WORLD,
    rank: int = 0,
    markers: Iterable[tuple[str, int]] = (('cells', 1, get_cell_tags), ),
    meshtags: bool = False,
    **gmsh_mesh_kwargs,
) -> Mesh | tuple[Mesh, MeshTags_int32, MeshTagsMetaClass]:
    model = annulus_model(
        centre,
        Rinner,
        Router, 
        Nr,
        name,
        cell,
        structured,
        comm,
        rank,
        markers,
        **gmsh_mesh_kwargs,
    )
    dim = 2
    return _model_to_mesh(model, comm, rank, dim, meshtags)


def annulus_model(
    centre: tuple[float, float],
    Rinner: float,
    Router: float, 
    Nr: int,
    name: str,
    cell: CellType = CellType.TRIANGLE,
    structured: bool = False, #TODO
    comm = MPI.COMM_WORLD,
    rank: int = 0,
    markers: Iterable[tuple[str, int]] = (('cells', 1, get_cell_tags), ),
    gmsh_options: dict | None = None,
    **gmsh_mesh_kwargs,
) -> gmsh.model:
    
    if not comm.rank == rank:
        return gmsh.model

    if not gmsh.isInitialized():
        print('initializing')
        gmsh.initialize()

    gmsh.model.add(name)
    centre = (*centre, 0)
    dim = 2

    # NOTE alternative creation of `annulus`
    # c_inner = gmsh.model.occ.addCircle(*centre, Rinner)
    # c_outer = gmsh.model.occ.addCircle(*centre, Router)
    # curve_inner = gmsh.model.occ.addCurveLoop([c_inner])
    # curve_outer = gmsh.model.occ.addCurveLoop([c_outer])
    # annulus = gmsh.model.occ.addPlaneSurface([curve_outer, curve_inner])
    # gmsh.model.occ.synchronize() 

    disk_outer = gmsh.model.occ.add_disk(*centre, Router, Router)
    disk_inner = gmsh.model.occ.add_disk(*centre, Rinner, Rinner)
    gmsh.model.occ.cut([(dim, disk_outer)], [(dim, disk_inner)])
    gmsh.model.occ.synchronize()

    for name, tag, get_tags in markers:
        dim, tags = get_tags(gmsh.model)
        gmsh.model.addPhysicalGroup(
            dim, tags, tag, name,
        )

    res = (Router - Rinner) / Nr
    ### TODO separate function
    _gmsh_mesh_kwargs = {
        "CharacteristicLengthMin": res,
        "CharacteristicLengthMax": res,
    }
    _gmsh_mesh_kwargs.update(gmsh_mesh_kwargs)
    for name, value in _gmsh_mesh_kwargs.items():
        gmsh.option.setNumber(f'Mesh.{name}', value)
    if cell == CellType.QUADRILATERAL:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

    if gmsh_options is None:
        gmsh_options = {}
    _gmsh_options = {"General.Verbosity": 0}
    _gmsh_options.update(gmsh_options)
    for name, value in _gmsh_options.items():
        gmsh.option.setNumber(name, value)
    ### 

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim)

    return gmsh.model



def ellipse(
    centre: tuple[float, float],
    radius: float,
    Nr: int,
    name: str,
    cell: CellType = CellType.TRIANGLE,
    comm = MPI.COMM_WORLD,
    rank: int = 0,
    markers: Iterable[tuple[str, int]] = (('cells', 1, get_cell_tags), ),
    meshtags: bool = False,
    **mesh_kwargs,
) -> Mesh | tuple[Mesh, MeshTags_int32, MeshTagsMetaClass]:
    model = ellipse_model()


def ellipse_model(
    centre: tuple[float, float],
    Rx: float,
    Ry: float,
    Nr: int,
) -> gmsh.model:
    raise NotImplementedError


def ellipsoid():
    ...


def ellipsoid_model(
    centre: tuple[float, float, float],
    Rx: float,
    Ry: float,
    Rz: float,
):
    ...

