from ufl.core.expr import Expr
from ufl import as_matrix, grad

from lucifex.fem import Function


def streamfunction_velocity(
    psi: Function,
    neg: bool = False,
) -> Expr:
    """
    `𝐮 = ((0, 1), (-1, 0))·∇ψ`
    """
    if neg:
        mat = ([0, -1], [1, 0])
    else:
        mat = ([0, 1], [-1, 0])
    return as_matrix(mat) * grad(psi)