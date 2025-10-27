import pytest

import numpy as np
from lucifex.fem import Function
from lucifex.mesh import rectangle_mesh
from lucifex.utils import minimum, maximum


@pytest.mark.parametrize("fs", [('P', 1), ('P', 2), ('DP', 1)])
def test_minimum_scalar(fs):
    mesh = rectangle_mesh(2.0, 1.0, 10, 10)
    u = Function((mesh, *fs))
    u.interpolate(lambda x: x[0] * x[1])
    umin = minimum(u)
    assert np.isclose(umin, 0.0)

@pytest.mark.parametrize("fs", [('P', 1), ('P', 2), ('DP', 1)])
@pytest.mark.parametrize("Lx", [1.0, 1.78, 2.0])
@pytest.mark.parametrize("Ly", [1.0, 1.23, 2.0])
def test_maximum_scalar(fs, Lx, Ly):
    mesh = rectangle_mesh(Lx, Ly, 10, 10)
    u = Function((mesh, *fs))
    u.interpolate(lambda x: x[0] * x[1])
    umax = maximum(u)
    assert np.isclose(umax, Lx * Ly)


@pytest.mark.parametrize("fs", [('P', 1, 2), ('P', 2, 2), ('BDM', 1)])
@pytest.mark.parametrize("Lx", [1.0, 1.78, 2.0])
@pytest.mark.parametrize("Ly", [1.0, 1.23, 2.0])
def test_minimum_vector(fs, Lx, Ly):
    mesh = rectangle_mesh(Lx, Ly, 10, 10)
    u = Function((mesh, *fs))
    u.interpolate(lambda x: (x[0], x[1]))
    umin = minimum(u)
    assert np.isclose(umin, 0.0)


@pytest.mark.parametrize("fs", [('P', 1, 2), ('P', 2, 2), ('BDM', 1)])
@pytest.mark.parametrize("Lx", [1.0, 1.78, 2.0])
@pytest.mark.parametrize("Ly", [1.0, 1.23, 2.0])
def test_maximum_vector(fs, Lx, Ly):
    mesh = rectangle_mesh(Lx, Ly, 10, 10)
    u = Function((mesh, *fs))
    u.interpolate(lambda x: (x[0], x[1]))
    umax = maximum(u)
    assert np.isclose(umax, np.sqrt(Lx**2 + Ly**2))