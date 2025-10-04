import gmsh

from ..utils.py_utils import replicate_callable
from .utils import create_gmsh_mesh_factory


def annulus_model(
    Rinner: float,
    Router: float, 
    centre: tuple[float, float] = (0.0, 0.0),
) -> gmsh.model:
    
    centre = (*centre, 0)
    dim = 2
    disk_outer = gmsh.model.occ.add_disk(*centre, Router, Router)
    disk_inner = gmsh.model.occ.add_disk(*centre, Rinner, Rinner)
    gmsh.model.occ.cut([(dim, disk_outer)], [(dim, disk_inner)])
    return gmsh.model


@replicate_callable(
    create_gmsh_mesh_factory(annulus_model, 2, 'annulus'),
)
def annulus_mesh():
    pass


def arc_model(
    radius: float,
    angle: float | tuple[float, float], 
) -> gmsh.model:
    if not isinstance(angle, tuple):
        angle = (0.0, angle)

    


def ellipse_model(
    radius: tuple[float, float] | float,
    centre: tuple[float, float] = (0.0, 0.0),
) -> gmsh.model:
    centre = (*centre, 0)
    if not isinstance(radius, tuple):
        radius = (radius, radius)
    gmsh.model.occ.add_disk(*centre, *radius)
    return gmsh.model


@replicate_callable(
    create_gmsh_mesh_factory(ellipse_model, 2, 'ellipse'),
)
def ellipse_mesh():
    pass


def ellipsoid_model(
) -> gmsh.model:
    raise NotImplementedError


@replicate_callable(
    create_gmsh_mesh_factory(ellipsoid_model, 3, 'ellipsoid'),
)
def ellipsoid_mesh():
    pass

