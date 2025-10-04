import gmsh
import numpy as np

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


def circle_sector_model(
    radius: float,
    angle: float | tuple[float, float], 
    centre: tuple[float, float] = (0.0, 0.0),
) -> gmsh.model:
    if not isinstance(angle, tuple):
        angle = (0.0, angle)
    angle = tuple(np.pi * i / 180 for i in angle)
    centre = (*centre, 0)
    start = (
        centre[0] + radius * np.cos(angle[0]), 
        centre[1] + radius * np.sin(angle[0]), 
        0,
    )
    end = (
        centre[0] + radius * np.cos(angle[1]), 
        centre[1] + radius * np.sin(angle[1]), 
        0,
    )

    centre = gmsh.model.geo.addPoint(*centre)
    start = gmsh.model.geo.addPoint(*start)
    end = gmsh.model.geo.addPoint(*end)

    arc = gmsh.model.geo.addCircleArc(start, centre, end)
    start_line = gmsh.model.geo.addLine(centre, start)        
    end_line = gmsh.model.geo.addLine(end, centre)    

    loop = gmsh.model.geo.addCurveLoop([start_line, arc, end_line])
    gmsh.model.geo.addPlaneSurface([loop])
    return gmsh.model

    
@replicate_callable(
    create_gmsh_mesh_factory(circle_sector_model, 2, 'circle_sector'),
)
def circle_sector_mesh():
    pass


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

