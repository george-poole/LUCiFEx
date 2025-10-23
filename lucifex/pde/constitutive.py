from ufl import Identity, sym, cos, sin, as_tensor
from ufl.core.expr import Expr


from lucifex.fdm.ufl_operators import nabla_grad
from lucifex.fem import LUCiFExFunction as Function, LUCiFExConstant as Constant


def strain(u: Function | Expr) -> Expr:
    """
    `ε(𝐮) = (∇𝐮 + ∇𝐮ᵀ) / 2`
    """
    return sym(nabla_grad(u))


def newtonian_stress(
    u: Function | Expr, 
    mu: Constant | float,
) -> Expr:
    """
    `𝜏(𝐮) = 2με(𝐮) = μ(∇𝐮 + ∇𝐮ᵀ)`
    """
    return 2 * mu * strain(u)


def permeability_cross_bedded(
    k: Function | Expr,
    kappa: Constant,
    vartheta: Constant,
):
    """
    `𝖪(ϕ) = K(ϕ) (
        (cos²ϑ + κsin²ϑ , (1 - κ)cosϑsinϑ), 
        ((1 - κ)cosϑsinϑ , κcos²ϑ + sin²ϑ), 
    )`
    """
    cs = cos(vartheta)
    sn = sin(vartheta)  
    tensor = as_tensor(
        (
            (cs**2 + kappa*sn**2, (1 - kappa)*cs*sn),
            ((1 - kappa)*cs*sn, kappa*cs**2 + sn**2), 
        ),
    )
    return k * tensor