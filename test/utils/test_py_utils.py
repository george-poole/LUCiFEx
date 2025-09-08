import pytest

from lucifex.utils.py_utils import (
    as_slice,
    filter_kwargs,
)

@pytest.mark.parametrize(
    "arg, expected",
    [
        (slice(0, None), slice(0, None)),
        (slice(1,10,2), slice(1,10,2)),
        ('0:2', slice(0, 2)),
        ('0:100:3', slice(0, 100, 3)),
        (':', slice(0, None)),
        (':5', slice(0, 5)),
        ('::2', slice(0, None, 2))
    ]
)
def test_as_slice(arg, expected):
    assert as_slice(arg) == expected


def test_filter_kwargs():
    func = lambda x, y, z: (x, y, z)
    kwargs = dict(x=1, y=2, z=3)
    assert filter_kwargs(func)(**kwargs) == tuple(kwargs.values())
    assert filter_kwargs(func)(**kwargs, w=4) == tuple(kwargs.values())