from enum import Enum

from dolfinx.mesh import CellType as DolfinxCellType
from dolfinx.mesh import DiagonalType as DolfinxDiagonalType


# TODO just Enum instead of str, Enum?
class StrEnum(str, Enum):

    def __repr__(self) -> str:
        return repr(self.value)


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
    """Enumeration class of implemented boundary condition types"""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"
    ANTIPERIODIC = "antiperiodic"
    ESSENTIAL = "essential"
    NATURAL = "natural"