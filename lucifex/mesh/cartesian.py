from typing import Literal

from mpi4py import MPI
from dolfinx.mesh import Mesh
from dolfinx.mesh import create_rectangle, create_interval, create_box

from ..utils.enum_types import CellType, DiagonalType 


def interval_mesh(
    Lx: float | tuple[float, float],
    Nx: int,
    name: str | None = None,
    comm: MPI.Comm | str = MPI.COMM_WORLD,
    **kwargs,
) -> Mesh:
    if not isinstance(Lx, tuple):
        Lx = (0.0, Lx)
    if isinstance(comm, str):
        comm = getattr(MPI, comm)

    mesh = create_interval(comm, Nx, Lx, **kwargs)

    if name is not None:
        mesh.name = name

    return mesh


def rectangle_mesh(
    Lx: float | tuple[float, float],
    Ly: float | tuple[float, float],
    Nx: int,
    Ny: int,
    name: str | None = None,
    cell: CellType | DiagonalType = CellType.QUADRILATERAL,
    comm: MPI.Comm | str = MPI.COMM_WORLD,
    **kwargs,
) -> Mesh:
    if not isinstance(Lx, tuple):
        Lx = (0.0, Lx)
    if not isinstance(Ly, tuple):
        Ly = (0.0, Ly)
    if isinstance(comm, str):
        comm = getattr(MPI, comm)

    bottom_left = (Lx[0], Ly[0])
    top_right = (Lx[1], Ly[1])

    if cell == CellType.QUADRILATERAL:
        cell = CellType(cell)
        mesh = create_rectangle(
        comm, (bottom_left, top_right), (Nx, Ny), cell.cpp_type, **kwargs,)
    else:
        if cell == CellType.TRIANGLE:
            diagonal = DiagonalType.RIGHT
        else:
            diagonal = DiagonalType(cell)
        mesh = create_rectangle(
            comm, 
            (bottom_left, top_right), 
            (Nx, Ny), 
            CellType.TRIANGLE.cpp_type, 
            diagonal=diagonal.cpp_type, 
            **kwargs,
        )

    if name is not None:
        mesh.name = name

    return mesh


def box_mesh(
    Lx: float | tuple[float, float],
    Ly: float | tuple[float, float],
    Lz: float | tuple[float, float],
    Nx: int,
    Ny: int,
    Nz: int,
    name: str | None = None,
    cell: CellType = CellType.HEXAHEDRON,
    comm: MPI.Comm | str = MPI.COMM_WORLD,
    **kwargs,
) -> Mesh:
    if not isinstance(Lx, tuple):
        Lx = (0.0, Lx)
    if not isinstance(Ly, tuple):
        Ly = (0.0, Ly)
    if not isinstance(Lz, tuple):
        Lz = (0.0, Lz)
    if isinstance(comm, str):
        comm = getattr(MPI, comm)

    cell = CellType(cell)
    bottom_left = (Lx[0], Ly[0], Lz[0])
    top_right = (Lx[1], Ly[1], Lz[1])

    mesh = create_box(
            comm, 
            (bottom_left, top_right), 
            (Nx, Ny, Nz), 
            cell.cpp_type, 
            **kwargs,
        )

    if name is not None:
        mesh.name = name

    return mesh
