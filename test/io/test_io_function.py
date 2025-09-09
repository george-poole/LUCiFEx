import pytest
import os

from lucifex.fem import LUCiFExFunction as Function
from lucifex.mesh import rectangle_mesh
from lucifex.io import write, read

DIR_PATH = os.path.dirname(__file__)

@pytest.mark.parametrize("Nx", [2, 5, 10])
@pytest.mark.parametrize("Ny", [2, 5, 10])
@pytest.mark.parametrize("elem", [('DP', 0), ('P', 1), ('P', 2)])
def test_io_scalar_function(Nx, Ny, elem):
    file_name = test_io_scalar_function.__name__
    u_name = f'u{elem[0]}{elem[1]}'
    mesh = rectangle_mesh(1.0, 1.0, Nx, Ny)
    u = Function((mesh, *elem), name=u_name)
    write(u, file_name, DIR_PATH, mode='w')
    elem_io = elem if elem == ('DP', 0) else ('P', 1)
    u_io = Function((mesh, *elem_io), name=u.name)
    read(u_io, DIR_PATH, file_name)
    if elem_io == ('DP', 0):
        assert len(u_io.x.array) == Nx * Ny
    else:
        assert len(u_io.x.array) == (Nx + 1) * (Ny + 1)


@pytest.mark.parametrize("Nx", [2, 5, 10])
@pytest.mark.parametrize("Ny", [2, 5, 10])
@pytest.mark.parametrize("elem", [('DP', 0), ('P', 1), ('P', 2)])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_io_vector_function(Nx, Ny, elem, dim):
    file_name = test_io_vector_function.__name__
    mesh = rectangle_mesh(1.0, 1.0, Nx, Ny)
    u = Function((mesh, *elem, dim), name='u')
    write(u, file_name, DIR_PATH, mode='w')
    elem_io = ('DP', 0) if elem == ('DP', 0) and dim == mesh.geometry.dim else ('P', 1)
    u_io = Function((mesh, *elem_io, dim), name=u.name)
    read(u_io, DIR_PATH, file_name)
    if elem_io == ('DP', 0):
        assert len(u_io.x.array) == Nx * Ny * dim
    else:
        assert len(u_io.x.array) == (Nx + 1) * (Ny + 1) * dim