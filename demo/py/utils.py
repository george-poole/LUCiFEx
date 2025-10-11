from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
from ufl import FacetNormal, inner, grad

from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.mesh import MeshBoundary, rectangle_mesh, interval_mesh, mesh_boundary


def interval_domain(
    Lx: float,
    Nx: int,
    name: str = 'Lx',
    names: tuple[str, str, str, str] = ('left', 'right'),
) -> tuple[Mesh, MeshBoundary]:
    mesh = interval_mesh(Lx, Nx, name)
    boundary = mesh_boundary(
        mesh,
        {
            names[0]: lambda x: x[0],
            names[1]: lambda x: x[1] - Lx,
        },
    )
    return mesh, boundary


def rectangle_domain(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    cell: str,
    name: str = 'LxLy',
    clockwise_names: tuple[str, str, str, str] = ('upper', 'right', 'lower', 'left'),
) -> tuple[Mesh, MeshBoundary]:
    mesh = rectangle_mesh(Lx, Ly, Nx, Ny, name, cell)
    boundary = mesh_boundary(
        mesh,
        {
            clockwise_names[0]: lambda x: x[1] - Ly,
            clockwise_names[1]: lambda x: x[0] - Lx,
            clockwise_names[2]: lambda x: x[1],
            clockwise_names[3]: lambda x: x[0],
        },
    )
    return mesh, boundary


def flux(
    c: Function,
    u: Function | Constant,
    d: Function,
    Pe: Constant
) -> tuple[Expr, Expr]:
    flux_adv = inner(u, grad(c))
    n = FacetNormal(c.function_space.mesh)
    flux_diff =  (1/Pe) * inner(n, d * grad(c))
    return flux_adv, flux_diff
