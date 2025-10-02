from typing import Iterable, Callable
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
    return cells[0][0], [cells[0][1]] 
    

def _setup_model(
    model: gmsh.model,
    name: str,
) -> None:
    if not gmsh.isInitialized():
        gmsh.initialize()
    model.add(name)


def _set_model_options(
    h: float,  
    cell: str,
    gmsh_options: dict, 
    gmsh_mesh_kwargs: dict,
) -> None:
    if cell == CellType.QUADRILATERAL:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

    _gmsh_mesh_kwargs = {
        "CharacteristicLengthMin": h,
        "CharacteristicLengthMax": h,
    }
    _gmsh_mesh_kwargs.update(gmsh_mesh_kwargs)
    for name, value in _gmsh_mesh_kwargs.items():
        gmsh.option.setNumber(f'Mesh.{name}', value)

    if gmsh_options is None:
        gmsh_options = {}
    _gmsh_options = {"General.Verbosity": 0}
    _gmsh_options.update(gmsh_options)
    for name, value in _gmsh_options.items():
        gmsh.option.setNumber(name, value)


def _set_model_groups(
    model: gmsh.model,
    markers: Iterable[tuple[str, int, Callable]],
) -> None:
    for name, tag, get_tags in markers:
        dim, tags = get_tags(model)
        model.addPhysicalGroup(
            dim, tags, tag, name,
        )
    

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
    Rinner: float,
    Router: float, 
    Nr: int,
    name: str,
    centre: tuple[float, float] = (0.0, 0.0),
    cell: CellType = CellType.TRIANGLE,
    structured: bool = False, #TODO
    comm = MPI.COMM_WORLD,
    rank: int = 0,
    markers: Iterable[tuple[str, int, Callable]] = (('cells', 1, partial(get_entity_tags, dim=2)), ),
    meshtags: bool = False,
    gmsh_options: dict | None = None,
    **gmsh_mesh_kwargs,
) -> Mesh | tuple[Mesh, MeshTags_int32, MeshTagsMetaClass]:
    model = annulus_model(
        Rinner,
        Router, 
        Nr,
        name,
        centre,
        cell,
        structured,
        comm,
        rank,
        markers,
        gmsh_options
        **gmsh_mesh_kwargs,
    )
    dim = 2
    return _model_to_mesh(model, comm, rank, dim, meshtags)


def annulus_model(
    Rinner: float,
    Router: float, 
    Nr: int,
    name: str,
    centre: tuple[float, float] = (0.0, 0.0),
    cell: CellType = CellType.TRIANGLE,
    structured: bool = False,
    comm = MPI.COMM_WORLD,
    rank: int = 0,
    markers: Iterable[tuple[str, int]] = (('cells', 1, partial(get_entity_tags, dim=2)), ),
    gmsh_options: dict | None = None,
    **gmsh_mesh_kwargs,
) -> gmsh.model:
    
    if not comm.rank == rank:
        return gmsh.model
    _setup_model(gmsh.model, name)

    centre = (*centre, 0)
    dim = 2

    if structured:
        raise NotImplementedError  #TODO
    else:
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
    _set_model_groups(gmsh.model, markers)
    dr = (Router - Rinner) / Nr
    _set_model_options(dr, cell, gmsh_options, gmsh_mesh_kwargs)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim)
    return gmsh.model


def ellipse_mesh(
    Rx: float,
    Ry: float, 
    Nr: int,
    name: str,
    centre: tuple[float, float] = (0.0, 0.0),
    cell: CellType = CellType.TRIANGLE,
    comm = MPI.COMM_WORLD,
    rank: int = 0,
    markers: Iterable[tuple[str, int]] = (('cells', 1, partial(get_entity_tags, dim=2)), ),
    meshtags: bool = False,
    gmsh_options: dict | None = None,
    **gmsh_mesh_kwargs,
) -> Mesh | tuple[Mesh, MeshTags_int32, MeshTagsMetaClass]:
    model = ellipse_model(
        Rx,
        Ry,
        Nr,
        name,
        centre,
        cell,
        comm,
        rank,
        markers,
        gmsh_options,
        **gmsh_mesh_kwargs,
    )
    dim = 2
    return _model_to_mesh(model, comm, rank, dim, meshtags)


def ellipse_model(
    Rx: float,
    Ry: float, 
    Nr: int,
    name: str,
    centre: tuple[float, float] = (0.0, 0.0),
    cell: CellType = CellType.TRIANGLE,
    comm = MPI.COMM_WORLD,
    rank: int = 0,
    markers: Iterable[tuple[str, int]] = (('cells', 1, partial(get_entity_tags, dim=2)), ),
    gmsh_options: dict | None = None,
    **gmsh_mesh_kwargs,
) -> gmsh.model:
    if not comm.rank == rank:
        return gmsh.model
    _setup_model(gmsh.model, name)

    centre = (*centre, 0)
    dim = 2

    gmsh.model.occ.add_disk(*centre, Rx, Ry)
    gmsh.model.occ.synchronize()

    _set_model_groups(gmsh.model, markers)
    dr = min(Rx, Ry) / Nr
    _set_model_options(dr, cell, gmsh_options, gmsh_mesh_kwargs)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim)
    return gmsh.model


def ellipsoid_mesh(
) -> Mesh:
    raise NotImplementedError 


def ellipsoid_model(
) -> gmsh.model:
    raise NotImplementedError

