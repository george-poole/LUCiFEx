import gmsh

from ..utils.py_utils import replicate_callable
from .utils import create_gmsh_mesh_factory


def ellipse_obstacle_model(
    Lx: float | tuple[float, float],
    Ly: float | tuple[float, float],
    radius: float | tuple[float, float],
    centre: tuple[float, float] | None = None,
) -> gmsh.model:
    if not isinstance(Lx, tuple):
        Lx = (0.0, Lx)
    if not isinstance(Ly, tuple):
        Ly = (0.0, Ly)
    if not isinstance(radius, tuple):
        radius = (radius, radius)
    
    dx = Lx[1] - Lx[0]
    dy = Ly[1] - Ly[0]
    if centre is None:
        centre = (Lx[0] + dx / 2, Ly[0] + dy / 2)
    centre = (*centre, 0)

    rectangle = gmsh.model.occ.addRectangle(Lx[0], Ly[0], 0, dx, dy)
    ellipse = gmsh.model.occ.addDisk(*centre, *radius)
    gmsh.model.occ.cut([(2, rectangle)], [(2, ellipse)])

    return gmsh.model
    

@replicate_callable(
    create_gmsh_mesh_factory(ellipse_obstacle_model, 2, 'ellipse_obstacle')
)
def ellipse_obstacle_mesh():
    pass


def anticline_model(
) -> gmsh.model :
    ...


@replicate_callable(
    create_gmsh_mesh_factory(anticline_model, 2, 'anticline')
)
def anticline_mesh():
    pass