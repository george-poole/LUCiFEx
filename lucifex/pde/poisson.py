from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, dx, ds, TestFunction, TrialFunction,
    inner, grad, TestFunction, TrialFunction, FacetNormal,
)
from ufl.geometry import GeometricCellQuantity

from dolfinx.fem import FunctionSpace
from lucifex.fem import Function, Constant
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions


def poisson(
    u: Function | FunctionSpace,
    f: Function | Constant | Expr,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇²u = f`
    """
    if isinstance(u, FunctionSpace):
        fs = u
    else:
        fs = u.function_space
    v = TestFunction(fs)
    u_trial = TrialFunction(fs)
    F_lhs = -inner(grad(v), grad(u_trial)) * dx
    F_rhs = -inner(v, f) * dx
    forms = [F_lhs, F_rhs]
    if bcs is not None:
        ds, u_neumann = bcs.boundary_data(u.function_space, 'neumann')
        F_neumann = sum([v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)
    return forms


def poisson_weak(
    u: Function | FunctionSpace,
    f: Function | Constant | Expr,
    h: str | GeometricCellQuantity, 
    alpha: float | Constant,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇²u = f`
    """
    forms = poisson(u, f, bcs=None)  

    if isinstance(u, FunctionSpace):
        fs = u
    else:
        fs = u.function_space
    mesh = fs.mesh

    v = TestFunction(fs)
    u_trial = TrialFunction(fs)
    n = FacetNormal(mesh)
    if isinstance(alpha, (float, int)):
        alpha = Constant(mesh, alpha)

    a_nistche = - v * u * inner(n, grad(u_trial)) * ds
    a_nistche += -inner(n, grad(v)) * u * ds 
    a_nistche += alpha / h * inner(u_trial, v) * ds
    forms.append(a_nistche)

    ds, u_neumann, u_dirichlet = bcs.boundary_data(u.function_space, 'neumann', 'dirichlet')

    if u_neumann:
        F_neumann = sum([v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)

    if u_dirichlet:
        l_nitsche = sum([-inner(n, grad(v)) * uD * ds(i) for i, uD in u_dirichlet])
        l_nitsche += sum([alpha / h * inner(uD, v) * ds(i) for i, uD in u_dirichlet])
        forms.append(l_nitsche)

    return forms


