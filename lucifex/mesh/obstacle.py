import gmsh

from ..utils.py_utils import replicate_callable
from .gmsh_utils import create_gmsh_mesh_factory
from ..utils import ToDoError


def rectangle_minus_ellipse_model(
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
    create_gmsh_mesh_factory(rectangle_minus_ellipse_model, 2, 'rectangle_minus_ellipse')
)
def rectangle_minus_ellipse_mesh():
    pass


def rectangle_minus_rectangle_model(
) -> gmsh.model :
    raise ToDoError


@replicate_callable(
    create_gmsh_mesh_factory(rectangle_minus_rectangle_model, 2, 'rectangle_minus_rectangle')
)
def rectangle_minus_rectangle_mesh():
    pass