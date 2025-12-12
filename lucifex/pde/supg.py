from typing import Callable
from functools import wraps

import numpy as np
from ufl import dx, Form, conditional, sqrt, conditional, lt, tanh, tr
from ufl.core.expr import Expr
from ufl.geometry import GeometricCellQuantity

from lucifex.fem import Function, Constant
from lucifex.fdm import (
    BE, FiniteDifference, FiniteDifferenceArgwise, Series, 
    ConstantSeries, ImplicitDiscretizationError,
)
from lucifex.fdm.ufl_operators import inner, grad
from lucifex.utils import is_tensor, is_vector, extract_mesh, cell_size_quantity
from lucifex.utils.py_utils import StrEnum


class TauType(StrEnum):
    CODINA = 'codina'
    SHAKIB = 'shakib'
    COTH = 'coth'
    COTH_APPROX = 'coth_approx'
    UPWIND = 'upwind'
    TRANSIENT = 'transient'


def supg_stabilization(
    supg: str | Callable,
    v: Expr,
    res: Expr,
    h: str | GeometricCellQuantity,
    a: Function | Constant | Series,
    d: Function | Constant | Series, 
    r: Function | Constant | Series = None,
    dt: Constant | ConstantSeries | None = None,
    D_adv: FiniteDifference | None = None,
    D_diff: FiniteDifference | None = None,
    D_reac: FiniteDifference | None = None,
    D_dt: FiniteDifference | None = None,
    phi = 1,
) -> Form:
    """
    `𝜏 (∇v·𝐚ᵉᶠᶠ)ℛ`
    """
    if isinstance(dt, ConstantSeries):
        dt = dt[0]

    if callable(supg):
        tau = supg(*[arg for arg in (h, a, d, r, dt) if arg is not None])
    else:
        if r is None:
            r = 0
        if D_adv is not None:
            a_eff = (1 / phi) * effective_velocity(a, D_adv, d, D_diff)
        if D_diff is not None:
            d_eff = (1 / phi) *  effective_diffusivity(d, D_diff)
        if D_reac is not None:
            r_eff = (1 / phi) * effective_reaction(r, D_reac, dt, D_dt)
        else:
            r_eff = r

        match supg:
            case TauType.CODINA:
                tau = tau_codina(h, a_eff, d_eff, r_eff)
            case TauType.SHAKIB:
                tau = tau_shakib(h, a_eff, d_eff, r_eff)
            case TauType.COTH:
                tau = tau_coth(h, a_eff, d_eff)
            case TauType.TRANSIENT:
                r_eff = (1 / phi) * effective_reaction(r, D_reac)
                tau = tau_transient(h, a_eff, d_eff, r_eff, dt)
            case TauType.UPWIND:
                tau = tau_upwind(h, a_eff)
            case _:
                raise ValueError(f"'{supg}' SUPG method not implemented.")
            
    return tau * inner(grad(v), a_eff) * res * dx


def tau_function(func):
    @wraps(func)
    def _(h, a, d, *args, **kwargs):
        mesh = extract_mesh(a)
        if isinstance(h, str):
            h = cell_size_quantity(mesh, h)
        if is_vector(a):
            a = sqrt(inner(a, a))
        if is_tensor(d):
            d = tr(d) / mesh.geometry.dim
        return func(h, a, d, *args, **kwargs)
    return _


@tau_function
def tau_codina(h, a, d, r) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = (2|𝐚| / h  +  4D / h²  -  R)⁻¹`
    """
    return ((2 * a / h) + (4 * d / h**2) - r) ** (-1) 


@tau_function
def tau_shakib(h, a, d, r) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = ( (2|𝐚| / h)²  +  9(4D / h²)²  +  R²)⁻¹ᐟ²`
    """
    return ((2 * a / h)**2 + 9 * (4 * d / h**2)**2 + r**2) ** (-0.5) 


@tau_function
def tau_transient(h, a, d, r, dt) -> Expr:
    """
    `∂u/∂t + 𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = ( (2 / Δt)² + (2|𝐚| / h)²  +  (4D / h²)²  +  R²)⁻¹ᐟ²`
    """
    return ((2 / dt)**2 + (2 * a / h)**2 + (2 * d / h**2)**2 + r**2) ** (-0.5) 


@tau_function
def tau_coth(h, a, d) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = h / 2|𝐚| ξ(Pe)`

    where `Pe = |𝐚|h / 2D` and `ξ(Pe) = coth(Pe) - 1/Pe`.
    """
    return (0.5 * h / a) * xi(peclet(h, a, d))


@tau_function
def tau_coth_approx(h, a, d) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = h / 2|𝐚| ξ(Pe)`

    where `Pe = |𝐚|h / 2D` and `ξ(Pe)` is approximated.
    """
    return (0.5 * h / a) * xi_approx(peclet(h, a, d))


@tau_function
def tau_upwind(h, a) -> Expr:
    """
    `𝐚·∇u = ∇·(D∇u) + Ru + s` \\
    `⟹ 𝜏 = h / 2|𝐚|`
    """
    return (0.5 * h / a) 


def peclet(h, a, d) -> Expr | float:
    """
    `Pe = |𝐚|h / 2D`
    """
    return 0.5 * a * h / d


def xi(Pe):
    """
    `ξ(Pe) = coth(Pe) - 1/Pe`
    """
    return 1/tanh(Pe) - 1/Pe


def xi_approx(Pe):
    """
    `ξ(Pe) = coth(Pe) - 1/Pe` \\
    `≈ Pe / 3` if `Pe < 3` \\
    `≈ 1` otherwise
    """
    return conditional(lt(Pe, 3), Pe / 3, 1)
    

def effective_velocity(
    a: Series, 
    D_adv: FiniteDifference | FiniteDifferenceArgwise,
    d: Function | Constant | None = None,
    D_diff: FiniteDifference | None = None,
) -> Expr:
    """
    `𝒟(𝐚·∇u) - ∇·(D𝒟(∇u)) = 𝐚ᵉᶠᶠ·∇uⁿ⁺¹ + ...` \\
    """
    if isinstance(D_adv, FiniteDifference):
        if D_adv.is_explicit:
            raise ImplicitDiscretizationError(D_adv, 'Advection term must be implicit w.r.t. transported quantity')
        a_eff = BE(a) * D_adv.explicit_coeff
    else:
        D_adv_a, D_adv_u = D_adv
        if D_adv_u.is_explicit:
            raise ImplicitDiscretizationError(D_adv_u, 'Advection term must be implicit w.r.t. transported quantity')
        a_eff = D_adv_a(a) * D_adv_u.explicit_coeff

    if d is not None:
        if D_diff is None:
            a_eff -= grad(d)
        else:
            a_eff -= grad(d) * D_diff.explicit_coeff

    return a_eff


def effective_diffusivity(
    d: Function | Constant,
    D_diff: FiniteDifference,
) -> Expr:
    """
    `∇·(D𝒟(∇u)) = Dᵉᶠᶠ ∇²uⁿ⁺¹ + ...`
    """
    d_eff = d * D_diff.explicit_coeff
    return d_eff


def effective_reaction(
    r: Series | Function | Constant,
    D_reac: FiniteDifference | FiniteDifferenceArgwise,
    dt: Constant | None = None,
    D_dt: FiniteDifference | None = None,
):
    """
    `∂u/∂t - 𝒟(Ru) = Rᵉᶠᶠ uⁿ⁺¹ + ...`
    """
    if isinstance(D_reac, FiniteDifference):
        r_eff = -BE(r) * D_reac.explicit_coeff
    else:
        D_reac_r, D_reac_u = D_reac
        r_eff = -D_reac_r(r) * D_reac_u.explicit_coeff

    if dt is not None:
        if D_dt is None:
            r_eff += 1 / dt
        else:
            r_eff += (1 / dt) * D_dt.explicit_coeff

    return r_eff


# TODO Lenardic1993 steps?
def limits_corrector(
    lower: float | None = None,
    upper: float | None = None,
    conservation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[[np.ndarray], None]:
    """
    Enforces `u ∈ [umin, umax]`

    NOTE intended for use on DoFs that are pointwise evaluations (e.g. Lagrange elements)
    """
    def _(u: np.ndarray) -> None:
        if conservation:
            mass_pre = conservation(np.copy(u))

        if lower is not None:
            u[u < lower] = lower
        if upper is not None:
            u[u > upper] = upper
        
        if conservation:
            mass_post = conservation(np.copy(u))
            mass_len = len([i for i in u if i not in (lower, upper)])
            u[:] += (mass_post - mass_pre) / mass_len
    
    return _