from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from ufl.core.expr import Expr
import numba
import numpy as np

from ..utils.fenicsx_utils import (
    create_function, dofs,
    mesh_vertices, mesh_axes,
    is_scalar, NonScalarError,
)
from ..utils.py_utils import optional_lru_cache