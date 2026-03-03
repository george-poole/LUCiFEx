from enum import Enum


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