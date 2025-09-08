import pytest

from lucifex.utils.deferred import defer, DeferredCondition

@pytest.mark.parametrize("arg", [1, 'one', (1.23, 4.56)])
def test_defer(arg):
    func = lambda x: x
    func_delayed = defer(func)
    assert func_delayed(arg)() == arg


def test_deferred_condition_with_mutable():
    x = [0]
    deferred = DeferredCondition(lambda: x[0] > 0)
    assert deferred.evaluate() is False
    x[0] += 1
    assert deferred.evaluate() is True
    x[0] -= 1
    assert deferred.evaluate() is False


def test_deferred_condition_with_arg():
    deferred = DeferredCondition(lambda x: x > 0)
    assert deferred.evaluate(-1) is False
    assert deferred.evaluate(0) is False
    assert deferred.evaluate(1) is True