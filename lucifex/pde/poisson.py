from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, Measure, TestFunction, TrialFunction,
    inner, grad, TestFunction, TrialFunction, FacetNormal,
)
from ufl.geometry import GeometricCellQuantity

from dolfinx.fem import FunctionSpace
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import is_none, cell_size_quantity
from lucifex.utils.py_utils import AnyFloat


def poisson(
    u: Function | FunctionSpace,
    f: Function | Constant | Expr | None = None,
    w: Function | Constant | Expr | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇²u = f` or `∇·(w·∇u) = f`
    """
    if w is None:
        w = 1

    if isinstance(u, FunctionSpace):
        fs = u
    else:
        fs = u.function_space

    dx = Measure('dx', fs.mesh)
    v = TestFunction(fs)
    u_trial = TrialFunction(fs)
    
    F_lhs = -inner(grad(v), w * grad(u_trial)) * dx
    forms = [F_lhs]

    if not is_none(f):
        F_rhs = -inner(v, f) * dx
        forms.append(F_rhs)

    if bcs is not None:
        ds, u_neumann, u_robin = bcs.boundary_data(u, 'neumann', 'robin')
        if u_neumann:
            F_neumann = sum([v * uN * ds(i) for i, uN in u_neumann])
            forms.append(F_neumann)
        if u_robin:
            F_robin = sum([v * uR * ds(i) for i, uR in u_robin])
            forms.append(F_robin)

    if is_none(f) and bcs is None:
        F_zero = v * Constant(fs.mesh, 0.0) * dx
        forms.append(F_zero)

    return forms


def nitsche_poisson(
    u: Function | FunctionSpace,
    f: Function | Constant | Expr,
    alpha: float | Constant,
    bcs: BoundaryConditions | None = None,
    h: str | GeometricCellQuantity = 'hdiam', 
) -> list[Form]:
    """
    `∇²u = f` with boundary conditions imposed by the Nitsche method.
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
    if isinstance(alpha, AnyFloat):
        alpha = Constant(mesh, alpha)
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)

    ds, u_neumann, u_dirichlet = bcs.boundary_data(u, 'neumann', 'dirichlet')

    a_nistche = - v * u * inner(n, grad(u_trial)) * ds
    a_nistche += -inner(n, grad(v)) * u * ds 
    a_nistche += alpha / h * inner(u_trial, v) * ds
    forms.append(a_nistche)

    if u_neumann:
        F_neumann = sum([v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)

    if u_dirichlet:
        l_nitsche = sum([-inner(n, grad(v)) * uD * ds(i) for i, uD in u_dirichlet])
        l_nitsche += sum([alpha / h * inner(uD, v) * ds(i) for i, uD in u_dirichlet])
        forms.append(l_nitsche)

    return forms


