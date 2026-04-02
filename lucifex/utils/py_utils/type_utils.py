from enum import Enum
from typing import TypeAlias

import numpy as np


AnyFloat: TypeAlias = float | np.floating
"""
Type alias to `float | np.floating`
"""

AnyInt: TypeAlias = int | np.integer
"""
Type alias to `int | np.integer`
"""

AnyNumber: TypeAlias = AnyFloat | AnyInt
"""
Type alias to `AnyFloat | AnyInt`
"""

AnyBool: TypeAlias = bool | np.bool_
"""
Type alias to `bool | np.bool_`
"""


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.name}'


class FloatEnum(float, Enum):
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.name}'
    
            
class ToDoError(NotImplementedError):
    def __init__(self):
        super().__init__('Working on it! Coming soon...')