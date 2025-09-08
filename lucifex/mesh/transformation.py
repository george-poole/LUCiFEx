from typing import Callable, Iterable

import dolfinx as dfx
import numpy as np

from ..utils import is_structured
from .cartesian import CellType
from .utils import overload_mesh


@overload_mesh
def transform(
    mesh: dfx.mesh.Mesh,
    func: Callable,
    index: int,
    strict: bool = False,
    bounds: bool = False, 
) -> None:
    
    assert is_transformable(mesh, strict)

    def _transform(x):
        x_copy = np.copy(x)
        x_copy[:, index] = func(x[:, index])
        return x_copy

    x = mesh.geometry.x
    x_transform = _transform(x)

    if not len(x_transform) == len(x):
        raise ValueError('Expected array of equal length after transform.')
    
    if bounds and not is_bounded_by_original(x_transform, x):
        raise ValueError('Expected array of equal bounds after transform.')

    mesh.geometry.x[:, :] = x_transform


def is_transformable(
    mesh: dfx.mesh.Mesh,
    strict: bool = True,
) -> bool:
    """Checks that the mesh is suitable for coordinate
    transformations"""

    if not is_structured(mesh):
       return False
    
    if not strict:
        return True
    else:
        dim = mesh.geometry.dim
        cell_name = mesh.topology.cell_name()
        match dim:
            case 1:
                return True
            case 2:
                return cell_name == CellType.QUADRILATERAL
            case 3:
                return cell_name == CellType.HEXAHEDRON
            

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
            
