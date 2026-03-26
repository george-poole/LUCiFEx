from collections.abc import Iterable
from typing_extensions import Self

from dolfinx.mesh import Mesh
from dolfinx.fem import Constant as DOLFINxConstant
import numpy as np

from ..utils.py_utils.str_utils import str_indexed
from .unsolved import UnsolvedType


class Constant(DOLFINxConstant):
    def __init__(
        self,
        mesh: Mesh,
        value: float | Iterable[float] | UnsolvedType | Self | None = None,
        name: str | None = None,
        shape: tuple[int, ...] = (),
        index: int | None = None,
    ):
        """
        Subclass with additional utilities, not to be confused with `dolfinx.fem.Constant`.
        A constant with respect to a mesh, but not necessarily with respect to time.
        """
        self._mesh = mesh

        if isinstance(name, tuple):
            name, subnames = name
            subnames = tuple(subnames)
        else:
            subnames = None
        if name is None:
            name =  f'{self.__class__.__name__}{id(self)}'

        self._name = name
        self._subnames = subnames
        self._create_subname = lambda i: (
            self._subnames[i] if self._subnames else f'{self.name}{i}'
        )

        if value is None:
            value = 0.0
        if isinstance(value, (int, UnsolvedType)):
            value = float(value)
        if not isinstance(value, Iterable):
            value = np.full(shape, value)
        if isinstance(value, DOLFINxConstant):
            value = value.value
        super().__init__(mesh, value)
        
        self._index = index

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n):
        assert isinstance(n, str)
        self._name = n

    def copy(self) -> Self:
        c = Constant(self.mesh, self.value.copy())
        c.name = self.name
        return c
    
    def sub(
        self, 
        index: int, 
        name: str | None = None,
    ) -> Self:
        if name is None:
            name = self._create_subname(index)
        return Constant(
            self.mesh,
            self.value[index],
            name,
            index=self.index
        )
    
    def split(
        self,
        names: tuple[str, ...] | None = None,
    ) -> tuple[Self, ...]:
        dim = self.ufl_shape[0]
        if names is None:
            names = [None] * dim
        return tuple(self.sub(i, n) for i, n in zip(range(dim), names, strict=True))
        
    
    @property
    def index(self) -> int | None:
        return self._index

    def __str__(self) -> str:
        s = self.name
        if self._index is None:
            return s
        else:
            return str_indexed(s, self._index, 'superscript', True)
        
    def __bool__(self) -> bool:
        return not bool(np.all(self.value == 0))