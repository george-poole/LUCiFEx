from ufl.core.expr import Expr
from ufl import (
    Form, inner, grad, Measure, TestFunction, TrialFunction,
    inner, grad, TestFunction, TrialFunction, cos,
    SpatialCoordinate,
)

from dolfinx.fem import FunctionSpace
from lucifex.fem import Function, Constant
from lucifex.solver import BoundaryConditions
from lucifex.utils.fenicsx_utils import extract_function_space

from .cg_transport import diffusion_forms


def helmholtz(
    u: Function | FunctionSpace,
    k: Constant | float | None = None,
    f: Function | Constant | Expr | None = None,
    d: Function | Constant | Expr | None = None,
    bcs: BoundaryConditions | None = None,
) -> tuple[Form, Form] | list[Form]:
    """
    `∇·(D·∇u) + k²u = f`

    If `k=None` and `f=None`, returns forms for the left and right hand sides
    of the eigenvalue problem `∇²u = λu` where `λ = -k²`.

    Otherwise returns forms for a boundary value problem.
    """
    if d is None:
        d = 1

    fs = extract_function_space(u)
    dx = Measure('dx', fs.mesh)
    v = TestFunction(fs)
    u_trial = TrialFunction(fs)

    F_eig = v * u_trial * dx
    F_lapl, *F_bcs = diffusion_forms(v, u_trial, 1, bcs=bcs, dx=dx)

    if k is None and f is None and not F_bcs:
        return F_lapl, F_eig
    else:
        F_eig = k**2 * F_eig
        F_src =  -v * f * dx
        return [F_lapl, F_eig, F_src, *F_bcs]
    

def mathieu(
    u: Function | FunctionSpace,
    q: Constant,
    k: Constant,
    bcs: BoundaryConditions | None = None,
) -> tuple[Form, Form] | tuple[Form, Form, Form]:
    """
    `-∇²u + 2qu = λu`
    """
    if isinstance(u, FunctionSpace):
        fs = u
    else:
        fs = u.function_space
    dx = Measure('dx', fs.mesh)
    v = TestFunction(fs)
    u_trial = TrialFunction(fs)
    x = SpatialCoordinate(fs.mesh)
    F_lapl = inner(grad(v), grad(u_trial)) * dx
    F_lapl += 2 * v * q * cos(inner(k, x)) * u_trial * dx
    F_eig = v * u_trial * dx

    if bcs is not None:
        ds, u_neumann = bcs.boundary_data(u, 'neumann')
        F_neumann = sum([-v * uN * ds(i) for i, uN in u_neumann])
        return F_lapl, -F_eig, F_neumann
    else:
        return F_lapl, F_eig