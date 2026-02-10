from typing import Any

from ufl.core.expr import Expr as DOLFINxExpr

from ..utils.py_utils.str_utils import str_indexed


class Expr(DOLFINxExpr):

    __class__ = DOLFINxExpr

    def __init__(
        self, 
        expr: DOLFINxExpr,
        name: str | None = None,
        index: int | None = None,
    ):
        self._expr = expr
        self._name = name
        self._index = index

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n):
        assert isinstance(n, str)
        self._name = n

    @property
    def index(self) -> int | None:
        return self._index
    
    def __str__(self) -> str:
        name = self.name
        if self.index is None:
            return name
        else:
            return str_indexed(name, self.index, 'superscript', True)
        
    def __repr__(self) -> str:
        return repr(self._expr)

    def __getattr__(self, name):
        return getattr(self._expr, name)
    
    def __add__(self, other):
        return self.__arithmetic(other, '__add__')
    
    def __radd__(self, other):
        return self.__arithmetic(other, '__radd__')
    
    def __sub__(self, other):
        return self.__arithmetic(other, '__sub__')
    
    def __rsub__(self, other):
        return self.__arithmetic(other, '__rsub__')
    
    def __mul__(self, other):
        return self.__arithmetic(other, '__mul__') 
    
    def __rmul__(self, other):
        return self.__arithmetic(other, '__rmul__')
    
    def __truediv__(self, other):
        return self.__arithmetic(other, '__truediv__')
    
    def __rtruediv__(self, other):
        return self.__arithmetic(other, '__rtruediv__')
    
    def __pow__(self, other):
        return self.__arithmetic(other, '__pow__')
    
    def __neg__(self, other):
        return self.__arithmetic(other, '__neg__')
    
    def __arithmetic(self, other: Any, method):
        return getattr(self._expr, method)(other)

    


