from ufl.core.expr import Expr
from ufl import as_matrix, grad

from lucifex.fem import Function


def streamfunction_velocity(psi: Function) -> Expr:
    """
    `𝐮 = ((0, 1), (-1, 0))·∇ψ`
    """
    return as_matrix([[0, 1], [-1, 0]]) * grad(psi)