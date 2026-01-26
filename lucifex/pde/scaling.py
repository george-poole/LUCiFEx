from typing import overload, Callable, ParamSpec, Generic
from typing_extensions import Unpack

from dolfinx.mesh import Mesh

from lucifex.fem import Constant


class ScalingMap:

    def __init__(
        self, 
        d: dict[str, float | int],
        name: str | None = None
    ):
        self._d = d
        self._name = name

    @property
    def map(self) -> dict[str, float | int]:
        return self._d
    
    @property
    def name(self) -> str | None:
        return self._name
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.map)}, '{self.name}')"

    @overload
    def __getitem__(
        self, 
        name: str, 
    ) -> int | float:
        ...

    @overload
    def __getitem__(
        self, 
        name: tuple[Mesh, str], 
    ) -> Constant:
        ...

    @overload
    def __getitem__(
        self, 
        name: tuple[Mesh, Unpack[tuple[str, ...]]], 
    ) -> Constant | tuple[Constant, ...]:
        ...

    @overload
    def __getitem__(
        self, 
        name: tuple[str, ...], 
    ) -> tuple[int | float, ...]:
        ...

    def __getitem__(
        self, 
        arg,
    ):
        if isinstance(arg, tuple):
            if not any(isinstance(i, Mesh) for i in arg):
                return tuple(self._d[i] for i in arg)
            else:
                mesh = arg[0]
                if len(arg) == 2:
                    i = arg[1]
                    return Constant(mesh, self._d[i], i)
                else:
                    return tuple(Constant(mesh, self._d[i], i) for i in arg[1:])
        else:
            return self._d[arg]
        

P = ParamSpec('P')
class ScalingOptions(Generic[P]):
    def __init__(
        self,
        symbols: tuple[str, ...],
        options: Callable[P, dict[str, tuple[float, ...]]],
    ):
        self._symbols = symbols
        self._options = options

    def __getitem__(
        self, 
        option: str,
    ) -> Callable[P, ScalingMap]:
        def _(*args: P.args, **kwargs: P.kwargs) -> ScalingMap:
            values = self._options(*args, **kwargs)[option]
            return ScalingMap(dict(zip(self._symbols, values)), option)
        return _
    