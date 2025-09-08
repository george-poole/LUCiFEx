import pytest

import numpy as np
from lucifex.fdm import cfl_timestep
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant
from lucifex.mesh import rectangle_mesh
from lucifex.utils import maximum, cell_sizes


@pytest.mark.parametrize("h", ['hmin', 'hmax'])
def test_cfl_timestep_h_str(h):
    mesh = rectangle_mesh(2.0, 1.0, 10, 10)
    u = Function((mesh, 'DP', 0, 2))
    u.interpolate(lambda x: (x[0], x[1]))
    dt = cfl_timestep(u, h)
    dt_expected = np.min(cell_sizes(mesh, h)) / maximum(u)
    assert np.isclose(dt, dt_expected)


@pytest.mark.parametrize("h", [0.01, 0.1, 0.123, 0.5])
def test_cfl_timestep_h_float(h):
    mesh = rectangle_mesh(2.0, 1.0, 10, 10)
    u = Function((mesh, 'DP', 0, 2))
    u.interpolate(lambda x: (x[0], x[1]))
    dt = cfl_timestep(u, h)
    dt_expected = h / maximum(u)
    assert np.isclose(dt, dt_expected)


@pytest.mark.parametrize("ux", [0.5, 1.25, -1.75])
@pytest.mark.parametrize("uy", [0.5, 1.25, -1.75])
@pytest.mark.parametrize("h", ['hmin', 'hmax'])
def test_cfl_timestep_u_constant(ux, uy, h):
    mesh = rectangle_mesh(2.0, 1.0, 10, 10)
    u = Constant(mesh, (ux, uy))
    dt = cfl_timestep(u, h)
    dt_expected = np.min(cell_sizes(mesh, h)) / maximum(u)
    assert np.isclose(dt, dt_expected)
    
