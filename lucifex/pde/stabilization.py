from typing import Callable, TypeAlias
from functools import wraps

from ufl import Argument, Measure, Form, div, sqrt, conditional, lt, tanh, tr
from ufl.core.expr import Expr
from ufl.geometry import GeometricCellQuantity

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    BE, DT, FiniteDifference, FiniteDifferenceArgwise, Series, 
    ConstantSeries, ExplicitDiscretizationError, peclet,
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.utils.fenicsx_utils import (
    is_tensor, is_vector, extract_mesh, 
    MeshExtractionError, cell_size_quantity,
)
from lucifex.utils.py_utils import StrEnum


class TauType(StrEnum):
    CODINA = 'codina'
    SHAKIB = 'shakib'
    COTH = 'coth'
    COTH_PWL = 'coth_pwl'
    COTH_INFTY = 'coth_infty'
    TRANSIENT = 'transient'
    NONE = 'none'
    UNSTABILIZED = 'unstabilized'


V: TypeAlias = Argument
A: TypeAlias = Expr | Function | Constant
D: TypeAlias = Expr | Function | Constant
R: TypeAlias = Expr | Function | Constant
def stabilization_form(
    tau: str | Callable,
    v: Expr,
    res: Expr,
    h: str | GeometricCellQuantity,
    a: Function | Constant | Series,
    d: Function | Constant | Series, 
    r: Function | Constant | Series = None,
    dt: Constant | ConstantSeries | None = None,
    D_adv: FiniteDifference = BE,
    D_diff: FiniteDifference = BE,
    D_reac: FiniteDifference = BE,
    D_dt: FiniteDifference = DT,
    phi: Expr = 1,
    Pv: str | Callable[[V, A, D, R], Expr] | None = None,
    dx: Measure = 1,
) -> Form:
    """
    Stabilisation term `𝜏 P(v; 𝐚ᵉᶠᶠ, Dᵉᶠᶠ, Rᵉᶠᶠ)ℛ` with stabilization
    parameter `𝜏` given some function `P(v)` of the test function and the 
    residual `ℛ`.
    """
    if isinstance(dt, ConstantSeries):
        dt = dt[0]

    if r is None:
        r = 0
    a_eff = (1 / phi) * effective_velocity(a, D_adv, d, D_diff)
    d_eff = (1 / phi) *  effective_diffusivity(d, D_diff)
    r_eff = (1 / phi) * effective_reaction(r, D_reac, dt, D_dt)

    match tau:
        case TauType.CODINA:
            tau = tau_codina(h, a_eff, d_eff, r_eff)
        case TauType.SHAKIB:
            tau = tau_shakib(h, a_eff, d_eff, r_eff)
        case TauType.TRANSIENT:
            r_eff = (1 / phi) * effective_reaction(r, D_reac)
            tau = tau_transient(h, a_eff, d_eff, r_eff, dt)
        case TauType.COTH:
            tau = tau_coth(h, a_eff, d_eff)
        case TauType.COTH_INFTY:
            tau = tau_coth_infty(h, a_eff)
        case TauType.COTH_PWL:
            tau = tau_coth_pwl(h, a_eff, d_eff)
        case TauType.NONE | TauType.UNSTABILIZED:
            tau = 0
        case _ if callable(tau):
            tau = tau(h, a_eff, d_eff, r_eff)
        case _:
            raise ValueError(
                f"Option '{tau}' for the stabilization parameter `𝜏` is not implemented."
            )
            
    match Pv:
        case 'supg' | None:
            Pv = Pv_supg(v, a_eff)
        case 'gls':
            Pv = Pv_gls(v, a_eff, d_eff, r_eff)
        case _ if callable(tau):
            Pv = Pv(v, a_eff, d_eff, r_eff)
        case _:
            raise ValueError(
                f"Option '{Pv}' for the function `P` is not implemented."
            )

    return tau * Pv * res * dx


def Pv_supg(v, a):
    """
    `P(v; 𝐚ᵉᶠᶠ) = ∇v·𝐚ᵉᶠᶠ`
    """
    return inner(grad(v), a)


def Pv_gls(v, a, d, r):
    return inner(grad(v), a) - div(d * grad(v)) - v * (r if r is not None else 0)


def tau_expr(func):
    """
    Decorator for functions returning an expression 
    for the stabilization parameter `𝜏 = 𝜏(h, 𝐚ᵉᶠᶠ, Dᵉᶠᶠ, Rᵉᶠᶠ, Δt)`.
    """
    @wraps(func)
    def _(
        h: GeometricCellQuantity | str, 
        a: Expr | Function | Constant, 
        *args, 
        **kwargs,
    ):
        mesh = None
        for i in (h, a, *args, kwargs.values()):
            try:
                if isinstance(i, Expr):
                    mesh = extract_mesh(i)
                    break
            except MeshExtractionError:
                pass

        if isinstance(h, str):
            if mesh is None :
                raise ValueError('Could not deduce mesh from arguments')
            h = cell_size_quantity(mesh, h)
        if is_vector(a):
            a = sqrt(inner(a, a))
        if args:
            d = args[0]
            if is_tensor(d):
                if mesh is None :
                    raise ValueError('Could not deduce mesh from arguments')    
                d = tr(d) / mesh.geometry.dim
            args = (d, *args[1:])
        return func(h, a, *args, **kwargs)
    return _


@tau_expr
def tau_codina(h, a, d, r) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = (2|𝐚| / h  +  4D / h²  -  R)⁻¹`
    """
    return ((2 * a / h) + (4 * d / h**2) - r) ** (-1) 


@tau_expr
def tau_shakib(h, a, d, r) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = ( (2|𝐚| / h)²  +  9(4D / h²)²  +  R²)⁻¹ᐟ²`
    """
    return ((2 * a / h)**2 + 9 * (4 * d / h**2)**2 + r**2) ** (-0.5) 


@tau_expr
def tau_transient(h, a, d, r, dt) -> Expr:
    """
    `∂u/∂t + 𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = ( (2 / Δt)² + (2|𝐚| / h)²  +  (4D / h²)²  +  R²)⁻¹ᐟ²`
    """
    return ((2 / dt)**2 + (2 * a / h)**2 + (2 * d / h**2)**2 + r**2) ** (-0.5) 


@tau_expr
def tau_coth(h, a, d) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = h / 2|𝐚| ξ(Pe)`

    where `Pe = |𝐚|h / 2D` and `ξ(Pe) = coth(Pe) - 1/Pe`.
    """
    return (0.5 * h / a) * xi(peclet(h, a, d))


@tau_expr
def tau_coth_pwl(h, a, d) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = h / 2|𝐚| ξ(Pe)`

    where `Pe = |𝐚|h / 2D` and the piecewise linear approximation of
    `ξ(Pe) = coth(Pe) - 1/Pe` is used.
    """
    return (0.5 * h / a) * xi_pwl(peclet(h, a, d))


@tau_expr
def tau_coth_infty(h, a) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = h / 2|𝐚|` 
    
    which is the `Pe -> ∞` limit of `𝜏 = h / 2|𝐚| ξ(Pe)`.
    """
    return (0.5 * h / a) 
        

def xi(Pe):
    """
    `ξ(Pe) = coth(Pe) - 1/Pe`
    """
    return 1/tanh(Pe) - 1/Pe


def xi_pwl(Pe):
    """
    `ξ(Pe) = coth(Pe) - 1/Pe` \\
    `≈ Pe / 3` if `Pe < 3` \\
    `≈ 1` otherwise
    """
    return conditional(lt(Pe, 3), Pe / 3, 1)
    

def effective_velocity(
    a: Series | Expr, 
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    d: Function | Constant | None = None,
    D_diff: FiniteDifference | None = None,
    strict: bool = True,
) -> Expr:
    """
    `𝒟(𝐚·∇u) - ∇·(D𝒟(∇u)) = 𝐚ᵉᶠᶠ·∇uⁿ⁺¹ + other terms`
    """
    if isinstance(D_adv, FiniteDifference):
        if strict and D_adv.is_explicit:
            raise ExplicitDiscretizationError(D_adv)
        a_eff = BE(a) * D_adv.implicit_coeff
    else:
        D_adv_a, D_adv_u = D_adv
        if strict and D_adv_u.is_explicit:
            raise ExplicitDiscretizationError(D_adv_u)
        a_eff = D_adv_a(a) * D_adv_u.implicit_coeff

    if d is None:
        return a_eff
    
    if not isinstance(d, Series):
        a_eff -= grad(d)
    else:
        # FIXME
        if isinstance(D_diff, FiniteDifferenceArgwise):
            D_diff_d, D_diff_u = D_diff
            a_eff -= grad(D_diff_d(d)) * D_diff_u.implicit_coeff
        else:
            a_eff -= grad(D_diff(d)) * D_diff.implicit_coeff

    return a_eff


def effective_diffusivity(
    d: Series | Expr,
    D_diff: FiniteDifference | FiniteDifferenceArgwise,
    strict: bool = True,
) -> Expr:
    """
    `∇·(𝒟(D∇u)) = ∇·(Dᵉᶠᶠ∇uⁿ⁺¹) + other terms`
    """
    if isinstance(D_diff, FiniteDifference):
        if strict and D_diff.is_explicit:
            raise ExplicitDiscretizationError(D_diff)
        d_eff = BE(d) * D_diff.implicit_coeff
    else:
        D_diff_d, D_diff_u = D_diff
        if strict and D_diff_u.is_explicit:
            raise ExplicitDiscretizationError(D_diff)
        d_eff = D_diff_d(d) * D_diff_u.implicit_coeff

    return d_eff


def effective_reaction(
    r: Series | Function | Constant,
    D_reac: FiniteDifference | FiniteDifferenceArgwise,
    dt: Constant | None = None,
    D_dt: FiniteDifference | None = None,
):
    """
    `𝒟(Ru) - 𝒟(∂u/∂t) = Rᵉᶠᶠ uⁿ⁺¹ + other terms`
    """
    if isinstance(D_reac, FiniteDifference):
        r_eff = BE(r) * D_reac.implicit_coeff
    else:
        D_reac_r, D_reac_u = D_reac
        r_eff = D_reac_r(r) * D_reac_u.implicit_coeff

    if dt is not None:
        if D_dt is None:
            r_eff -= 1 / dt
        else:
            r_eff -= (1 / dt) * D_dt.implicit_coeff

    return r_eff