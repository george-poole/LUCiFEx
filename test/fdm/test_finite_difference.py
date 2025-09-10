import pytest

from lucifex.mesh import interval_mesh
from lucifex.fdm import CN, FE, BE, DT, AB2, AB1, FiniteDifference, FunctionSeries, finite_difference_order
from lucifex.fem import is_unsolved


@pytest.fixture
def u():
    mesh = interval_mesh(1.0, 10)
    return FunctionSeries((mesh, 'P', 1), name='u', order=3, ics=0.0)

def test_unsolved(u: FunctionSeries):
    assert is_unsolved(u[1])
    assert not is_unsolved(u[0])
    assert is_unsolved(u[-1])


def test_crank_nicolson(u: FunctionSeries):
    assert str(CN(u)) == str(0.5 * u.trialfunction + 0.5 * u[0])


def test_forward_euler(u: FunctionSeries):
    assert str(FE(u)) == str(1.0 * u[0])


def test_backward_euler(u: FunctionSeries):
    assert str(BE(u)) == str(1.0 * u.trialfunction)


def test_time_derivative(u: FunctionSeries):
    dt = 0.01
    assert str(DT(u, dt)) == str((1.0 * u.trialfunction + -1.0 * u[0]) / dt)


def test_adams_bashforth_1(u: FunctionSeries):
    assert str(AB1(u)) == str(FE(u))


def test_adams_bashforth_2(u: FunctionSeries):
    assert str(AB2(u)) == str(1.5 * u[0] + -0.5 * u[-1])


@pytest.mark.parametrize(
    "fdm, expected",
    [
        (AB1, 1),
        (AB2, 2),
        (CN, 1),
        (FE, 1),
        (BE, 0),
        (DT, 1),
        ((AB2, CN), 2),
    ]
)
def test_finite_difference_order(fdm: FiniteDifference, expected: int):
    assert finite_difference_order(fdm) == expected
