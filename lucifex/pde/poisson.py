from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, Measure, TestFunction, TrialFunction,
    inner, grad, TestFunction, TrialFunction, FacetNormal,
)
from ufl.geometry import GeometricCellQuantity

from dolfinx.fem import FunctionSpace
from lucifex.fem import Function, Constant
from lucifex.fdm import FunctionSeries
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import is_none, cell_size_quantity, extract_function_space
from lucifex.utils.npy_utils import AnyFloat

from .cg_transport import diffusion_forms


def poisson(
    u: Function | FunctionSeries | FunctionSpace,
    f: Function | Constant | Expr | None = None,
    d: Function | Constant | Expr | None = None,
    bcs: BoundaryConditions | None = None,
) -> list[Form]:
    """
    `∇²u = f` or `∇·(D·∇u) = f`
    """
    if d is None:
        d = 1

    fs = extract_function_space(u)
    dx = Measure('dx', fs.mesh)
    v = TestFunction(fs)
    u_trial = TrialFunction(fs)

    forms = diffusion_forms(v, u_trial, d, bcs=bcs, dx=dx)

    if not is_none(f):
        F_rhs = -inner(v, f) * dx
        forms.append(F_rhs)

    if is_none(f) and bcs is None:
        F_zero = v * Constant(fs.mesh, 0.0) * dx
        forms.append(F_zero)

    return forms


def nitsche_poisson(
    u: Function | FunctionSeries | FunctionSpace,
    f: Function | Constant | Expr,
    alpha: float | Constant,
    bcs: BoundaryConditions | None = None,
    h: str | GeometricCellQuantity = 'hdiam', 
) -> list[Form]:
    """
    `∇²u = f` with boundary conditions imposed by the Nitsche method.
    """
    forms = poisson(u, f, bcs=None)  

    fs = extract_function_space(u)
    mesh = fs.mesh

    v = TestFunction(fs)
    u_trial = TrialFunction(fs)
    n = FacetNormal(mesh)
    if isinstance(alpha, AnyFloat):
        alpha = Constant(mesh, alpha)
    if isinstance(h, str):
        h = cell_size_quantity(mesh, h)

    ds, u_neumann, u_dirichlet = bcs.boundary_data(u_trial, 'neumann', 'dirichlet')

    F_nistche = - v * inner(n, grad(u_trial)) * ds
    F_nistche += -inner(n, grad(v)) * u_trial * ds 
    F_nistche += alpha / h * inner(u_trial, v) * ds
    forms.append(F_nistche)

    if u_neumann:
        F_neumann = sum([v * uN * ds(i) for i, uN in u_neumann])
        forms.append(F_neumann)

    if u_dirichlet:
        l_nitsche = sum([-inner(n, grad(v)) * uD * ds(i) for i, uD in u_dirichlet])
        l_nitsche += sum([alpha / h * inner(uD, v) * ds(i) for i, uD in u_dirichlet])
        forms.append(l_nitsche)

    return forms


