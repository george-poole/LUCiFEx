from typing import Any, overload, Protocol, runtime_checkable, TypeAlias, Literal
from inspect import isfunction
from collections.abc import Callable, Iterable
from operator import add, mul
from itertools import product
from functools import reduce

import numpy as np
from dolfinx.fem import Function, Constant, FunctionSpace, Expression
from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator

from ..utils.fenicsx_utils import (
    BoundaryType, mesh_coordinates, mesh_vertices,
    create_function, create_function_space, set_function,
)


MultivariateCallable: TypeAlias = Callable[
    [np.ndarray[tuple[int, Literal[3]]], float], 
    np.ndarray[tuple[int, Literal[3]], float]
]
"""
e.g. `lambda x: x[0] * x[1]`
"""

UnivariateCallable: TypeAlias = Callable[
    [np.ndarray[tuple[int], float]], 
    np.ndarray[tuple[int], float]
] | Callable[[float], float]
"""
e.g. `lambda x: x * x`
"""


@runtime_checkable
class Perturbation(Protocol):

    def base(
        self,
        fs: FunctionSpace,
    ) -> Function:
        """
        Returns the base function `b(𝐱)` only
        """
        ...

    def noise(
        self,
        fs: FunctionSpace,
    ) -> Function:
        """
        Returns the noise function `N(𝐱)` only
        """
        ...

    def combine_base_noise(
        self,
        fs: FunctionSpace,
        base: Function | None = None,
        corrector: Callable[[np.ndarray], None] | None = None,
        combiner: Callable[[Any, Any], Any] | None = None,
    ) -> Function:
        """
        Returns the base state and noise combined `u(𝐱) = 𝒪(b(𝐱), N(𝐱))`. 

        Default operator is addition `u(𝐱) = b(𝐱) + N(𝐱)`.
        """ 
        ...


class DofsPerturbation:
    def __init__(
        self,
        base: Callable[[np.ndarray], np.ndarray]
        | Function
        | Constant
        | Expression
        | float
        | Iterable[float],
        seed: int,
        amplitude: float | tuple[float, float],
        n_freq: tuple[int, ...] | None = None,
        corrector: Callable[[np.ndarray], None] | None = None,
        combiner: Callable[[Any, Any], Any] = add,
    ):
        self._base = base
        if isinstance(amplitude, float):
            amplitude = (0.0, amplitude)
        self._amplitude = amplitude
        self._n_freq = n_freq
        self._rng = np.random.default_rng(seed)
        if corrector is None:
            corrector = lambda _: None
        self._corrector = corrector
        self._combiner = combiner

    def base(
        self,
        fs: FunctionSpace,
        corrector: Callable[[np.ndarray], None] | None = None,
    ) -> Function:
        if corrector is None:
            corrector = self._corrector
        f = create_function(fs, self._base)
        self._corrector(f.x.array)
        return f
        
    def noise(
        self,
        fs: FunctionSpace,
        n_freq: tuple[float, ...] | None = None,
        **interpolator_kwargs: Any,
    ) -> Function:
        if n_freq is None:
            n_freq = self._n_freq

        f = create_function(fs)

        if n_freq is None:
            noise_fine = self._rng.uniform(*self._amplitude, len(f.x.array))
        else:
            vs = mesh_vertices(fs.mesh)
            xs = mesh_coordinates(fs.mesh)
            x_lims = [(np.min(i), np.max(i)) for i in xs]
            x_coarse = [np.linspace(*lim, num=f) for lim, f in zip(x_lims, n_freq, strict=True)]
            noise_coarse = self._rng.uniform(*self._amplitude, n_freq)
            noise_fine = RegularGridInterpolator(x_coarse, noise_coarse, **interpolator_kwargs)(vs)
        
        set_function(f, noise_fine, dofs_indices=':')
        return f
    
    def combine_base_noise(
        self,
        fs: FunctionSpace,
        base: Function | None = None,
        corrector: Callable[[np.ndarray], None] | None = None,
        combiner: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        name: str | None = None,
    ) -> Function:
        if corrector is None:
            corrector = self._corrector
        if combiner is None:
            combiner = self._combiner

        fs = create_function_space(fs)
        if base is None:
            base = self.base(fs, False) 
        perturbation = self.noise(fs)

        f = create_function(fs, name=name)
        dofs = combiner(base.x.array, perturbation.x.array)
        set_function(f, dofs, dofs_indices=':')
        self._corrector(f.x.array)
        return f


class SpatialPerturbation:
    """
    Combines a scalar-valued base function `b(𝐱)` with a noise function `N(𝐱)` to produce 
    a perturbation 
    
    `b(𝐱) + ϵN(𝐱)`

    of amplitude `ϵ`.
    """
    def __init__(
        self,
        base: Callable[[np.ndarray], np.ndarray]
        | Function
        | Constant
        | Expression
        | float
        | Iterable[float],
        noise_factory: Callable[[np.ndarray], np.ndarray],
        bbox: list[float | tuple[float, float]] | np.ndarray,
        amplitude: float | tuple[float, float],
        corrector: Callable[[np.ndarray], None] | None = None, #FIXME subspace case,
        combiner: Callable[[Any, Any], Any] = add,
        **rescale_kwargs,
    ) -> None:
        self._base = base
        self._noise = rescale(noise_factory, bbox, amplitude, **rescale_kwargs)
        if corrector is None:
            corrector = lambda _: None
        self._corrector = corrector
        self._combiner = combiner

    def base(
        self,
        fs: FunctionSpace,
        correct: bool = True,
    ) -> Function:
        f = create_function(fs, self._base)
        if correct:
            self._corrector(f.x.array)
        return f

    def noise(
        self,
        fs: FunctionSpace,
    ) -> Function:
        return create_function(fs, self._noise)

    def combine_base_noise(
        self,
        fs: FunctionSpace,
        base: Function | None = None,
        corrector: Callable[[np.ndarray], None] | None = None,
        combiner: Callable[[Any, Any], Any] | None = None,
        name: str | None = None,
    ) -> Function: 
        if corrector is None:
            corrector = self._corrector
        if combiner is None:
            combiner = self._combiner

        if isfunction(self._base) and isfunction(self._noise):
            perturbed = lambda x: combiner(self._base(x), self._noise(x))
        else:
            if base is None:
                base = self.base(fs, False) 
            perturbation = self.noise(fs)
            perturbed = combiner(base, perturbation)
        f = create_function(fs, perturbed, name=name)
        self._corrector(f.x.array)
        return f
    

@overload
def cubic_noise(
    bc_type: str | tuple[str, str],
    bbox: float | tuple[float, float],
    n_freq: int,
    seed: int, 
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    ...


@overload
def cubic_noise(
    bc_type: list[str | tuple[str, str]],
    bbox: list[float | tuple[float, float]],
    n_freq: Iterable[int],
    seed: Iterable[int], 
    index: Iterable[int],
) -> Callable[[np.ndarray], np.ndarray]:
    ...
    

def cubic_noise(
    bc_type: str | tuple[str, str] | list[str | tuple[str, str]],
    bbox: float | tuple[float, float] | list[float | tuple[float, float]],
    n_freq: int | Iterable[int],
    seed: int | Iterable[int], 
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(bc_type, list):
        assert isinstance(bbox, list)
        assert isinstance(n_freq, Iterable)
        assert isinstance(seed, Iterable)
        assert not isinstance(index, int)
        if index is None:
            index = tuple(range(len(bc_type)))
        assert index is not None
        noises = [cubic_noise(b, l, f, s, i) for b, l, f, s, i in 
                  zip(bc_type, bbox, n_freq, seed, index, strict=True)]
        return lambda x: reduce(mul, [n(x) for n in noises])

    if isinstance(bc_type, str):
        bc_type = (bc_type, bc_type)
    bc_type = tuple(BoundaryType(i) for i in bc_type)
    
    if isinstance(bbox, float):
        bbox = (0, bbox)
    
    x_coarse = np.linspace(*bbox, num=n_freq)
    rng = np.random.default_rng(seed)
    noise_coarse = rng.uniform(0, 1, x_coarse.size)

    match bc_type:
        case (BoundaryType.PERIODIC, BoundaryType.PERIODIC):
            # `N(x₋) = N(x₊)`
            noise_coarse[0] = noise_coarse[-1]
            noise = CubicSpline(x_coarse, noise_coarse, bc_type="periodic")

        case (BoundaryType.NEUMANN, BoundaryType.NEUMANN):
            # `∂N/∂x(x₋) = ∂N/∂x(x₊) = 0`
            noise = CubicSpline(x_coarse, noise_coarse, bc_type="clamped")

        case (BoundaryType.DIRICHLET, BoundaryType.DIRICHLET):
            # `N(x₋) = N(x₊) = 0`
            noise_coarse[0] = 0.0
            noise_coarse[-1] = 0.0
            noise = PchipInterpolator(x_coarse, noise_coarse)

        case (BoundaryType.NEUMANN, BoundaryType.DIRICHLET):
            # `∂N/∂x(x₋) = N(x₊) = 0`
            noise_coarse[-1] = 0.0
            x_coarse = np.insert(x_coarse, 1, 0.5 * (x_coarse[0] + x_coarse[1]))
            noise_coarse = np.insert(noise_coarse, 1, noise_coarse[0])
            noise = PchipInterpolator(x_coarse, noise_coarse)

        case (BoundaryType.DIRICHLET, BoundaryType.NEUMANN):
            # `N(x₋) = ∂N/∂x(x₊) = 0`
            noise_coarse[0] = 0.0
            x_coarse = np.insert(x_coarse, -1, 0.5 * (x_coarse[-2] + x_coarse[-1]))
            noise_coarse = np.insert(noise_coarse, -1, noise_coarse[-1])
            noise = PchipInterpolator(x_coarse, noise_coarse)

        case _:
            raise ValueError(bc_type)
        
    if index is not None:
        return lambda x: noise(x[index])
    else:
        return lambda x: noise(x)


@overload
def sinusoid_noise(
    bc_type: str | tuple[str, str] | list[str | tuple[str, str]],
    bbox: float | tuple[float, float] | list[float | tuple[float, float]],
    n_waves: int | Iterable[int],
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    ...


@overload
def sinusoid_noise(
    bc_type: list[str | tuple[str, str]],
    bbox: list[float | tuple[float, float]],
    n_waves: Iterable[int],
    index: Iterable[int],
) -> Callable[[np.ndarray], np.ndarray]:
    ...


def sinusoid_noise(
    bc_type: str | tuple[str, str] | list[str | tuple[str, str]],
    bbox: float | tuple[float, float] | list[float | tuple[float, float]],
    n_waves: int | Iterable[int],
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(bc_type, list):
        assert isinstance(bbox, list)
        assert isinstance(n_waves, Iterable)
        assert not isinstance(index, int)
        if index is None:
            index = tuple(range(len(bc_type)))
        noises = [sinusoid_noise(b, l, w, i) for b, l, w, i in zip(bc_type, bbox, n_waves, index, strict=True)]
        return lambda x: reduce(mul, [n(x) for n in noises])

    if isinstance(bc_type, str):
        bc_type = (bc_type, bc_type)
    bc_type = tuple(BoundaryType(i) for i in bc_type)

    if isinstance(bbox, float):
        bbox = (0, bbox)

    x0 = bbox[0]
    Lx = bbox[1] - bbox[0]

    match bc_type:
        case (BoundaryType.PERIODIC, BoundaryType.PERIODIC):
            # `N(x₋) = N(x₊)` and anti-symmetric about centre
            wavelength = Lx / n_waves
            noise = lambda x: np.sin(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.DIRICHLET, BoundaryType.DIRICHLET):
            # `N(x₋) = N(x₊) = 0` and symmetric about centre
            wavelength = Lx / (n_waves + 0.5)
            noise = lambda x: np.sin(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.NEUMANN, BoundaryType.NEUMANN):
            # `∂N/∂x(x₋) = ∂N/∂x(x₊) = 0` and symmetric about centre
            wavelength = Lx / n_waves
            noise = lambda x: np.cos(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.NEUMANN, BoundaryType.DIRICHLET):
            # `∂N/∂x(x₋) = N(x₊) = 0`
            wavelength = Lx / (n_waves + 0.25)
            noise = lambda x: np.cos(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.DIRICHLET, BoundaryType.NEUMANN):
            # `N(x₋) = ∂N/∂x(x₊) = 0`
            wavelength = Lx / (n_waves + 0.25)
            noise = lambda x: np.sin(2 * np.pi * (x - x0) / wavelength)
        case _:
            raise ValueError(bc_type)
        
       
    if index is not None:
        return lambda x: noise(x[index])
    else:
        return lambda x: noise(x)
    

@overload
def rescale(
    clbl: Callable[[np.ndarray], np.ndarray],
    bbox: float | tuple[float, float],
    amplitude: float | tuple[float, float],
    n_fine: int = 1e6,
    eps: float = 1e-4,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Rescales a univariate function such as `lambda x: x`
    defined on the domain `x ∈ [x₋, x₊]` to the range `f ∈ [f₋, f₊]`.
    """
    ...

@overload
def rescale(
    clbl: Callable[[np.ndarray], np.ndarray],
    bbox: list[float | tuple[float, float]] | np.ndarray,
    amplitude: float | tuple[float, float],
    n_fine: int = 1e6,
    eps: float = 1e-4,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Rescales a multivariate function such as `lambda x: x[0] * x[1] * x[2]`
    defined on the domain `(x, y, z) ∈ [x₋, x₊] × [y₋, y₊] × [z₋, z₊]` 
    to the range `f ∈ [f₋, f₊]`.
    """
    ...


def rescale(
    clbl: Callable[[np.ndarray], np.ndarray],
    bbox: float | tuple[float, float] | list[float | tuple[float, float]] | np.ndarray,
    amplitude: float | tuple[float, float],
    n_fine: int = 1e4,
    eps: float = 1e-4,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    `n_fine >> 0` and `0 < eps << 1` for robustness
    """
    if isinstance(amplitude, float):
        amplitude = (0, amplitude)
    f_min, f_max = amplitude

    if not is_multivariate(clbl):
        if isinstance(bbox, float):
            bbox = (0, bbox)
        x_fine = np.linspace(*bbox, num=int(n_fine))
        noise_fine = clbl(x_fine)
    else:
        if isinstance(bbox, np.ndarray):
            x_fine_vertices = bbox
        else:
            if not isinstance(bbox, list):
                raise TypeError(f'Bounding box needs to be a list, not {bbox}.')
            bbox = [(0, i) if isinstance(i, float) else i for i in bbox]
            dim = len(bbox)
            x_fine_axes = [np.linspace(*d, num=int(n_fine ** (1/dim))) for d in bbox]
            x_fine_vertices = np.array([i for i in product(*x_fine_axes)])
        noise_fine = [clbl(x) for x in x_fine_vertices]

    noise_min = np.min(noise_fine)
    noise_max = np.max(noise_fine)
    denom = (1.0 + eps) * (noise_max - noise_min)
    unit_rescaled = lambda x: (clbl(x) - noise_min) / denom
    return lambda x: f_min + (f_max - f_min) * unit_rescaled(x)


def multivariate_noise(
    bbox: Iterable[float | tuple[float, float]],
    n_freq: Iterable[int],
    seed: int,
    **kwargs: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    bbox = [(0, i) if isinstance(i, float) else i for i in bbox]
    x_coarse = tuple(np.linspace(*i, num=n) for i, n in zip(bbox, n_freq, strict=True))
    noise_shape = tuple(i.shape for i in x_coarse)
    rng = np.random.default_rng(seed)
    noise_coarse = rng.uniform(0, 1, noise_shape)
    noise = RegularGridInterpolator(x_coarse, noise_coarse, **kwargs)
    return lambda x: noise(np.stack([x[1], x[0]], axis=-1)).flatten() # FIXME stacking 2d vs 3d


def is_multivariate(
    clbl: Callable[[np.ndarray], np.ndarray] | Callable[[float], float],
) -> bool:
    x_test = 0.0
    try:
        f = clbl(x_test)
        assert isinstance(f, float)
        return False
    except TypeError:
        return True
    
    
    