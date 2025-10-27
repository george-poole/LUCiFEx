import pytest

from lucifex.mesh import interval_mesh
from lucifex.fem import (
    Function, 
    Constant,
    is_unsolved, Unsolved,
)

@pytest.fixture
def mesh():
    return interval_mesh(1.0, 10)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        (1.23, False),
        (lambda x: x[0], False),
        (Unsolved, True),
        (Unsolved.value, True),
    ]
)
def test_function_is_unsolved(mesh, value, expected):
    u = Function((mesh, 'P', 1), value)
    assert is_unsolved(u) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        (1.23, False),
        ((1.23, 4.56), False),
        (Unsolved, True),
        (Unsolved.value, True),
        ((1.23, Unsolved.value), True),
    ]
)
def test_constant_is_unsolved(mesh, value, expected):
    c = Constant(mesh, value)
    assert is_unsolved(c) == expected