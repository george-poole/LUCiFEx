from typing import Callable, Iterable

import dolfinx as dfx
import numpy as np

from ..utils import is_structured
from .refine import is_simplex_mesh
from .utils import overload_mesh


@overload_mesh
def transform(
    mesh: dfx.mesh.Mesh,
    func: Callable,
    index: int,
    strict: bool = False,
    bounded: bool = False, 
) -> None:
    
    if strict:
        assert not is_simplex_mesh(mesh)
        assert is_structured(mesh)

    def _transform(x):
        x_copy = np.copy(x)
        x_copy[:, index] = func(x[:, index])
        return x_copy

    x = mesh.geometry.x
    x_transform = _transform(x)

    if not len(x_transform) == len(x):
        raise ValueError('Expected array of equal length after transform.')
    
    if bounded and not is_bounded_by_original(x_transform, x):
        raise ValueError('Expected array of equal bounds after transform.')

    mesh.geometry.x[:, :] = x_transform
            

def is_bounded_by_original(
    x_transform: np.ndarray, 
    x_original: np.ndarray,
    indices: Iterable[int] = (0, 1, 2),
) -> bool:
    for i in indices:
        xo_i = x_original[:, i]
        xt_i = x_transform[:, i]
        if not np.isclose(np.min(xo_i) , np.min(xt_i)):
            return False
        if not np.isclose(np.max(xo_i), np.max(xt_i)):
            return False
    return True
            
