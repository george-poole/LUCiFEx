from dolfinx.mesh import CellType as DolfinxCellType
from dolfinx.mesh import DiagonalType as DolfinxDiagonalType

from .py_utils import StrEnum


class CellType(StrEnum): 
    """Enumeration class containing implemented cell types"""

    TRIANGLE = "triangle"
    QUADRILATERAL = "quadrilateral"
    TETRAHEDRON = "tetrahedron"
    HEXAHEDRON = "hexahedron"
    
    @property
    def cpp_type(self):
        return getattr(DolfinxCellType, self.value)


class DiagonalType(StrEnum): 
    """Enumeration class containing implemented cell diagonal types"""

    LEFT = "left"
    RIGHT = "right"
    LEFT_RIGHT = "left_right"
    RIGHT_LEFT = "right_left"
    CROSSED = "crossed"
    
    @property
    def cpp_type(self):
        return getattr(DolfinxDiagonalType, self.value)
    

class BoundaryType(StrEnum): 
    """Enumeration class containing implemented boundary condition types"""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    ESSENTIAL = "essential"
    NATURAL = "natural"
    PERIODIC = "periodic"
    ANTIPERIODIC = "antiperiodic"
    WEAK_DIRICHLET = "weak_dirichlet"


class DofsMethodType(StrEnum):
    GEOMETRICAL = 'geometrical'
    TOPOLOGICAL = 'topological'