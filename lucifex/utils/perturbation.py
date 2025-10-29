from typing import Any, overload, Protocol, runtime_checkable
from inspect import isfunction
from collections.abc import Callable, Iterable
from operator import add, mul
from itertools import product
from functools import reduce

import numpy as np
from dolfinx.fem import Function, Constant, FunctionSpace, Expression
from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator

from .enum_types import BoundaryType
from .dofs_utils import as_dofs_setter, SpatialMarkerAlias
from .mesh_utils import mesh_coordinates, mesh_vertices
from .fem_typecast import finite_element_function, function_space
from .fem_mutate import set_finite_element_function


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
        operator: Callable[[Any, Any], Any] = add,
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
        freq: tuple[int, ...] | None = None,
        dofs_corrector: Callable[[Function], None] 
        | Iterable[tuple[SpatialMarkerAlias, float | Constant] | tuple[SpatialMarkerAlias, float | Constant, int]] 
        | None = None,
    ):
        self._base = base
        if isinstance(amplitude, float):
            amplitude = (0.0, amplitude)
        self._amplitude = amplitude
        self._freq = freq
        self._rng = np.random.default_rng(seed)
        self._dofs_corrector = as_dofs_setter(dofs_corrector)

    def base(
        self,
        fs: FunctionSpace,
        correct: bool = True,
    ) -> Function:
        f = finite_element_function(fs, self._base)
        if correct:
            self._dofs_corrector(f)
        return f
        
    def noise(
        self,
        fs: FunctionSpace,
        freq: tuple[float, ...] | None = None,
        method: str | None = None,
    ) -> Function:
        if freq is None:
            freq = self._freq
        fs = function_space(fs)

        f = Function(fs)
        if freq is None:
            dofs = self._rng.uniform(*self._amplitude, len(f.x.array))
        else:
            method = 'linear' if method is None else method
            vs = mesh_vertices(fs.mesh)
            xs = mesh_coordinates(fs.mesh)
            xlims = [(np.min(i), np.max(i)) for i in xs]
            x_coarse = [np.linspace(*lim, num=f) for lim, f in zip(xlims, freq, strict=True)]
            noise = self._rng.uniform(*self._amplitude, freq)
            dofs = RegularGridInterpolator(x_coarse, noise, method=method)(vs)
        
        set_finite_element_function(f, dofs, dofs_indices=':')
        return f
    
    def combine_base_noise(
        self,
        fs: FunctionSpace,
        base: Function | None = None,
        operator: Callable[[np.ndarray, np.ndarray], np.ndarray] = add,
        correct: bool = True,
        name: str | None = None,
    ) -> Function:
        fs = function_space(fs)
        if base is None:
            base = self.base(fs, False) 
        perturbation = self.noise(fs)

        f = Function(fs, name=name)
        dofs = operator(base.x.array, perturbation.x.array)
        set_finite_element_function(f, dofs, dofs_indices=':')
        if correct:
            self._dofs_corrector(f)
        return f


class SpatialPerturbation:
    """
    Combines a base function `b(𝐱)` with a noise function `N(𝐱)` to produce 
    a perturbation 
    
    `b(𝐱) + ϵN(𝐱)`

    of amplitude `ϵ`.
    
    NOTE for scalar-valued functions only
    """
    def __init__(
        self,
        base: Callable[[np.ndarray], np.ndarray]
        | Function
        | Constant
        | Expression
        | float
        | Iterable[float],
        noise_shape: Callable[[np.ndarray], np.ndarray],
        domain_lims: list[float | tuple[float, float]] | np.ndarray,
        amplitude: float | tuple[float, float],
        dofs_corrector: Callable[[Function], None] 
        | Iterable[tuple[SpatialMarkerAlias, float | Constant] | tuple[SpatialMarkerAlias, float | Constant, int]] 
        | None = None, #FIXME subspace case,
        **rescale_kwargs,
    ) -> None:
        self._base = base
        self._noise = rescale(noise_shape, domain_lims, amplitude, **rescale_kwargs)
        self._dofs_corrector = as_dofs_setter(dofs_corrector)

    def base(
        self,
        fs: FunctionSpace,
        correct: bool = True,
    ) -> Function:
        f = finite_element_function(fs, self._base)
        if correct:
            self._dofs_corrector(f)
        return f

    def noise(
        self,
        fs: FunctionSpace,
    ) -> Function:
        return finite_element_function(fs, self._noise)

    def combine_base_noise(
        self,
        fs: FunctionSpace,
        base: Function | None = None,
        operator: Callable[[Any, Any], Any] = add,
        correct: bool = True,
        name: str | None = None,
    ) -> Function:       
        if isfunction(self._base) and isfunction(self._noise):
            perturbed = lambda x: operator(self._base(x), self._noise(x))
        else:
            if base is None:
                base = self.base(fs, False) 
            perturbation = self.noise(fs)
            perturbed = operator(base, perturbation)
        f = finite_element_function(fs, perturbed)
        if correct:
            self._dofs_corrector(f)
        if name is not None:
            f.name = name
        return f
    

def random_noise(
    shape: int | tuple[int, ...],
    seed: int,
):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, shape)
    

@overload
def cubic_noise(
    boundary: str | tuple[str, str],
    interval: float | tuple[float, float],
    n_freq: int,
    seed: int, 
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    ...


@overload
def cubic_noise(
    boundary: list[str | tuple[str, str]],
    interval: list[float | tuple[float, float]],
    n_freq: Iterable[int],
    seed: Iterable[int], 
    index: Iterable[int],
) -> Callable[[np.ndarray], np.ndarray]:
    ...
    

def cubic_noise(
    boundary: str | tuple[str, str] | list[str | tuple[str, str]],
    interval: float | tuple[float, float] | list[float | tuple[float, float]],
    n_freq: int | Iterable[int],
    seed: int | Iterable[int], 
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(boundary, list):
        assert isinstance(interval, list)
        assert isinstance(n_freq, Iterable)
        assert isinstance(seed, Iterable)
        assert not isinstance(index, int)
        if index is None:
            index = tuple(range(len(boundary)))
        assert index is not None
        noises = [cubic_noise(b, l, f, s, i) for b, l, f, s, i in 
                  zip(boundary, interval, n_freq, seed, index, strict=True)]
        return lambda x: reduce(mul, [n(x) for n in noises])

    if isinstance(boundary, str):
        boundary = (boundary, boundary)
    if isinstance(interval, float):
        interval = (0, interval)
    
    x_coarse = np.linspace(*interval, num=n_freq)
    rng = np.random.default_rng(seed)
    noise_coarse = rng.uniform(0, 1, x_coarse.size)

    match boundary:
        case (BoundaryType.PERIODIC, BoundaryType.PERIODIC):
            # f(xmin) = f(xmax)
            # enforcing periodicity
            noise_coarse[0] = noise_coarse[-1]
            noise = CubicSpline(x_coarse, noise_coarse, bc_type="periodic")

        case (BoundaryType.NEUMANN, BoundaryType.NEUMANN):
            # dfdx(xmin) = 0 and dfdx(xmax) = 0
            noise = CubicSpline(x_coarse, noise_coarse, bc_type="clamped")

        case (BoundaryType.DIRICHLET, BoundaryType.DIRICHLET):
            # f(xmin) = 0 and f(xmax) = 0
            # enforcing Dirichlet boundary conditions
            noise_coarse[0] = 0.0
            noise_coarse[-1] = 0.0
            noise = PchipInterpolator(x_coarse, noise_coarse)

        case (BoundaryType.NEUMANN, BoundaryType.DIRICHLET):
            # dfdx(xmin) = 0 and f(xmax) = 0
            # enforcing Dirichlet boundary condition
            noise_coarse[-1] = 0.0
            # enforcing Neumann boundary condition by inserting an additional coarse value
            x_coarse = np.insert(x_coarse, 1, 0.5 * (x_coarse[0] + x_coarse[1]))
            noise_coarse = np.insert(noise_coarse, 1, noise_coarse[0])
            noise = PchipInterpolator(x_coarse, noise_coarse)

        case (BoundaryType.DIRICHLET, BoundaryType.NEUMANN):
            # f(xmin) = 0 and dfdx(xmax) = 0
            # enforcing Dirichlet boundary condition
            noise_coarse[0] = 0.0
            # enforcing Neumann boundary condition by inserting an additional coarse value
            x_coarse = np.insert(x_coarse, -1, 0.5 * (x_coarse[-2] + x_coarse[-1]))
            noise_coarse = np.insert(noise_coarse, -1, noise_coarse[-1])
            noise = PchipInterpolator(x_coarse, noise_coarse)

        case _:
            raise BoundaryTypeError(boundary)
        
    if index is not None:
        return lambda x: noise(x[index])
    else:
        return lambda x: noise(x)


@overload
def sinusoid_noise(
    boundary: str | tuple[str, str] | list[str | tuple[str, str]],
    interval: float | tuple[float, float] | list[float | tuple[float, float]],
    n_waves: int | Iterable[int],
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    ...


@overload
def sinusoid_noise(
    boundary: list[str | tuple[str, str]],
    interval: list[float | tuple[float, float]],
    n_waves: Iterable[int],
    index: Iterable[int],
) -> Callable[[np.ndarray], np.ndarray]:
    ...


def sinusoid_noise(
    boundary: str | tuple[str, str] | list[str | tuple[str, str]],
    interval: float | tuple[float, float] | list[float | tuple[float, float]],
    n_waves: int | Iterable[int],
    index: int | Iterable[int] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(boundary, list):
        assert isinstance(interval, list)
        assert isinstance(n_waves, Iterable)
        assert not isinstance(index, int)
        if index is None:
            index = tuple(range(len(boundary)))
        noises = [sinusoid_noise(b, l, w, i) for b, l, w, i in zip(boundary, interval, n_waves, index, strict=True)]
        return lambda x: reduce(mul, [n(x) for n in noises])

    if isinstance(boundary, str):
        boundary = (boundary, boundary)
    if isinstance(interval, float):
        interval = (0, interval)

    x0 = interval[0]
    Lx = interval[1] - interval[0]

    match boundary:
        case (BoundaryType.PERIODIC, BoundaryType.PERIODIC):
            # f(xmin) = f(xmax) and anti-symmetric about centre
            wavelength = Lx / n_waves
            noise = lambda x: np.sin(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.DIRICHLET, BoundaryType.DIRICHLET):
            # f(xmin) = 0 , f(xmax) = 0 and symmetric about centre
            wavelength = Lx / (n_waves + 0.5)
            noise = lambda x: np.sin(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.NEUMANN, BoundaryType.NEUMANN):
            # dfdx(xmin) = 0 , dfdx(xmax) = 0 and symmetric about centre
            wavelength = Lx / n_waves
            noise = lambda x: np.cos(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.NEUMANN, BoundaryType.DIRICHLET):
            # dfdx(xmin) = 0 and f(xmax) = 0
            wavelength = Lx / (n_waves + 0.25)
            noise = lambda x: np.cos(2 * np.pi * (x - x0) / wavelength)
        case (BoundaryType.DIRICHLET, BoundaryType.NEUMANN):
            # f(xmin) = 0 and dfdx(xmax) = 0
            wavelength = Lx / (n_waves + 0.25)
            noise = lambda x: np.sin(2 * np.pi * (x - x0) / wavelength)
        case _:
            raise BoundaryTypeError(boundary)
        
       
    if index is not None:
        return lambda x: noise(x[index])
    else:
        return lambda x: noise(x)
    

@overload
def rescale(
    func: Callable[[np.ndarray], np.ndarray],
    domain_lims: float | tuple[float, float],
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
    func: Callable[[np.ndarray], np.ndarray],
    domain_lims: list[float | tuple[float, float]] | np.ndarray,
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
    func: Callable[[np.ndarray], np.ndarray],
    domain_lims: float | tuple[float, float] | list[float | tuple[float, float]] | np.ndarray,
    amplitude: float | tuple[float, float],
    n_fine: int = 1e4,
    eps: float = 1e-4,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(amplitude, float):
        amplitude = (0, amplitude)
    f_min, f_max = amplitude

    if is_univariate(func):
        if isinstance(domain_lims, float):
            domain_lims = (0, domain_lims)
        # large num > 0 and small ε > 0 for robustness
        x_fine = np.linspace(*domain_lims, num=int(n_fine))
        noise_fine = func(x_fine)
    else:
        if isinstance(domain_lims, np.ndarray):
            x_fine_vertices = domain_lims
        else:
            assert isinstance(domain_lims, list)
            domain_lims = [(0, d) if isinstance(d, float) else d for d in domain_lims]
            dim = len(domain_lims)
            x_fine_axes = [np.linspace(*d, num=int(n_fine ** (1/dim))) for d in domain_lims]
            x_fine_vertices = np.array([i for i in product(*x_fine_axes)])
        noise_fine = [func(x) for x in x_fine_vertices]

    noise_min = np.min(noise_fine)
    noise_max = np.max(noise_fine)
    denom = (1.0 + eps) * (noise_max - noise_min)
    # rescaling to unit interval `f ∈ [0, 1]`
    unit_rescaled = lambda x: (func(x) - noise_min) / denom
    # rescaling to chosen interval `f ∈ [f₋, f₊]`
    rescaled = lambda x: f_min + (f_max - f_min) * unit_rescaled(x)
    return rescaled


def is_univariate(
    func: Callable[[np.ndarray], np.ndarray],
) -> bool:
    x_test = 0.0
    try:
        f = func(x_test)
        assert isinstance(f, float)
        return True
    except TypeError:
        return False
    

BoundaryTypeError = lambda b: ValueError(f'Boundary type {b} not valid.')
       

# def _multiply_noise(
#     noise: Callable,
#     limit: float | tuple[float, float],
#     *args: list | Iterable,
# ):
#     if isinstance(limit, float):
#         limit = (0, limit)
#     offset, amplitude = limit
#     dim = len(args[0])
#     noises = [noise(amplitude ** (1/dim), *a) for a in zip(*args, strict=True)]
#     return lambda x: offset + reduce(mul, [n(x[i]) for i, n in enumerate(noises)])


# def _typecast_noise_args(
#     limit: float | tuple[float, float],
#     boundary: str | tuple[str, str],
#     Lx: float | tuple[float, float],
# ) -> tuple[tuple[float, float], 
#            tuple[str, str], 
#            tuple[float, float],
#            ]:
#     if isinstance(limit, float):
#         limit = (0.0, limit)
#     if isinstance(boundary, str):
#         boundary = (boundary, boundary)
#     if isinstance(Lx, float):
#         Lx = (0.0, Lx)
#     return limit, boundary, Lx