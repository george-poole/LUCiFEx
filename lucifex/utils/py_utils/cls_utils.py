from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class FloatEnum(float, Enum):
    def __str__(self) -> str:
        return str(self.value)
    
            
class ToDoError(NotImplementedError):
    def __init__(self):
        super().__init__('Working on it! Coming soon...')